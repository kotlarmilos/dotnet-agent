import dotenv from 'dotenv';
import { Octokit } from '@octokit/rest';
import { graphql } from '@octokit/graphql';
import fs from 'fs/promises';
import fsSync from 'fs';
import path from 'path';
import pLimit from 'p-limit';
import pRetry from 'p-retry';
import { promisify } from 'util';
import cliProgress from 'cli-progress';
import settings from '../settings.json' assert { type: 'json' };
import fetch from 'node-fetch';

dotenv.config();
const TOKEN = process.env.GITHUB_TOKEN;
if (!TOKEN) throw new Error('Missing GITHUB_TOKEN');

const owner = settings.repository.replace("https://github.com/", "").split('/')[0];
const repo = settings.repository.replace("https://github.com/", "").split('/')[1];
const outDir = 'data/raw-data';
const limit = Number.MAX_SAFE_INTEGER;

const octokit = new Octokit({ auth: TOKEN });
const gh      = graphql.defaults({ headers: { authorization: `token ${TOKEN}` } });
const sleep   = promisify(setTimeout);

function log(level, msg) {
  console.log(`[${new Date().toISOString()}] [${level}] ${msg}`);
}

async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true });
}

async function getRateLimit() {
  const { data } = await octokit.rest.rateLimit.get();
  const core = data.resources.core;
  return {
    remaining: core.remaining,
    resetAt:   core.reset * 1000,
    window:    core.reset * 1000 - Date.now()
  };
}

function msToHMS(ms) {
  const s   = Math.ceil(ms/1000);
  const h   = Math.floor(s/3600);
  const m   = Math.floor((s%3600)/60);
  const sec = s%60;
  return `${h}h ${m}m ${sec}s`;
}

async function waitForRateLimitReset() {
  while (true) {
    const { remaining, resetAt } = await getRateLimit();
    if (remaining > 0) return;
    const ms = Math.max(resetAt - Date.now(), 0);
    log('WARN', `Rate limit hit; sleeping ${msToHMS(ms)}`);
    await sleep(ms + 1000);
  }
}

async function checkRateLimit(threshold = 10) {
  const { remaining, resetAt } = await getRateLimit();
  if (remaining < threshold) {
    const ms = Math.max(resetAt - Date.now(), 0);
    log('WARN', `Approaching rate limit (${remaining} left); sleeping ${msToHMS(ms)}`);
    await sleep(ms + 1000);
  }
}

async function withSmartRetry(fn) {
  return pRetry(fn, {
    retries: 5,
    factor: 2,
    minTimeout: 1000,
    maxTimeout: 30000,
    onFailedAttempt: e => log('WARN', `Attempt #${e.attemptNumber} failed: ${e.status || e.code || e.message}`),
    retry: e =>
      (/rate limit/i.test(e.message)) ||
      (e.status >= 500) ||
      ['ECONNRESET','ENOTFOUND'].includes(e.code)
  });
}

async function callGh(query, vars) {
  try {
    return await gh(query, vars);
  } catch (err) {
    if (/rate limit/i.test(err.message)) {
      await waitForRateLimitReset();
      return await gh(query, vars);
    }
    throw err;
  }
}

const STATE_FILE = path.join(outDir, 'state.json');
let state = { prsDone: [], diffsDone: [], errors: [] };
try {
  state = JSON.parse(fsSync.readFileSync(STATE_FILE, 'utf-8'));
} catch {}
function saveState() {
  fsSync.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
}

async function fetchConnection(owner, repo, prNumber, field, selection) {
  let cursor = null, nodes = [];
  do {
    const resp = await withSmartRetry(() =>
      callGh(
        `query($owner:String!,$repo:String!,$pr:Int!,$first:Int!,$after:String){
           repository(owner:$owner,name:$repo){
             pullRequest(number:$pr){
               ${field}(first:$first,after:$after){
                 pageInfo{ hasNextPage endCursor }
                 nodes{ ${selection} }
               }
             }
           }
         }`,
        { owner, repo, pr: prNumber, first: 100, after: cursor }
      )
    );
    const conn = resp.repository.pullRequest[field];
    nodes.push(...conn.nodes);
    cursor = conn.pageInfo.hasNextPage ? conn.pageInfo.endCursor : null;
  } while (cursor);
  return nodes;
}

async function fetchAllPRs(owner, repo, limit) {
  if (!state.cursor) {
    const repoInfo = await callGh(`query($owner:String!,$repo:String!){repository(owner:$owner,name:$repo){pullRequests{totalCount}}}`, { owner, repo });
    state.totalPRs = limit ? Math.min(limit, repoInfo.repository.pullRequests.totalCount) : repoInfo.repository.pullRequests.totalCount;
    state.cursor = null;
    state.processedPRs = 0;
    state.prsCompleted = [];
    state.prsPending = [];
    saveState();
  }
  const bar = new cliProgress.SingleBar({
    format: 'Fetching PRs |{bar}| {value}/{total} PRs',
    hideCursor: true
  }, cliProgress.Presets.shades_classic);
  bar.start(limit ? Math.min(limit, state.totalPRs) : state.totalPRs, state.processedPRs);
  while (state.processedPRs < limit && (state.cursor || state.processedPRs === 0)) {  
    const resp = await withSmartRetry(() =>
      callGh(
        `query($owner:String!,$repo:String!,$first:Int!,$after:String){
           repository(owner:$owner,name:$repo){
             pullRequests(first:$first,after:$after,orderBy:{field:CREATED_AT,direction:DESC}){
               pageInfo{ hasNextPage endCursor }
               nodes{
                 number title body createdAt closedAt mergedAt state author{login}
                 labels(first:10){ nodes{ name }}
                 headRefName additions deletions changedFiles
                 comments(first:100) { totalCount nodes { body createdAt } }
                 reviewThreads(first:100) { totalCount nodes { comments(first:20) { nodes { path diffHunk body createdAt }}}}
                 commits(first:100) { totalCount nodes { commit { oid message committedDate }}}
               }
             }
           }
         }`,
        { owner, repo, first: 100, after: state.cursor }
      )
    );
    const conn = resp.repository.pullRequests;
    for (const pr of conn.nodes) {
      const prFile = path.join(outDir, 'prs', `pr-${pr.number}.json`);
      if (!fsSync.existsSync(prFile)) {
        await fs.writeFile(prFile, JSON.stringify(pr, null, 2));
      if (pr.comments.totalCount == pr.comments.nodes.length
          && pr.reviewThreads.totalCount == pr.reviewThreads.nodes.length
          && pr.commits.totalCount == pr.commits.nodes.length) {
            state.prsCompleted.push(pr.number);
          }else {
            state.prsPending.push(pr.number);
          }
      }
    }
    state.processedPRs += conn.nodes.length;
    state.cursor = conn.pageInfo.hasNextPage ? conn.pageInfo.endCursor : null;
    saveState();
    bar.update(state.processedPRs);
  };
  bar.stop();
}

async function fetchMissingDetails(pr) {
  const outFile = path.join(outDir, 'prs', `pr-${pr}.json`);
  const prData = JSON.parse(await fs.readFile(outFile, 'utf-8'));

  const commentCount = prData.comments?.totalCount || 0;
  const reviewCount  = prData.reviewThreads?.totalCount || 0;
  const commitCount  = prData.commits?.totalCount || 0;

  const fetchedCommentCount = prData.comments?.nodes?.length || 0;
  const fetchedReviewCount  = prData.reviewThreads?.nodes?.length || 0;
  const fetchedCommitCount  = prData.commits?.nodes?.length || 0;

  if (fetchedCommentCount < commentCount) {
    prData.comments.nodes = await fetchConnection(owner, repo, pr.number, 'comments', 'body createdAt');
  }
  if (fetchedReviewCount < reviewCount) {
    prData.reviewThreads.nodes = await fetchConnection(owner, repo, pr.number, 'reviewThreads', 'comments(first:100) { nodes { path diffHunk body createdAt } }');
  }
  if (fetchedCommitCount < commitCount) {
    prData.commits.nodes = await fetchConnection(owner, repo, pr.number, 'commits', 'commit { oid message committedDate }');
  }

  await fs.writeFile(outFile, JSON.stringify(prData, null, 2));
}

async function fetchCommitDiff(owner, repo, sha) {
  await checkRateLimit();
  const url = `https://github.com/${owner}/${repo}/commit/${sha}.diff`;
  const res = await withSmartRetry(() => fetch(url));
  if (!res.ok) throw new Error(`Failed to fetch diff: ${res.status} ${res.statusText}`);
  return await res.text();
}

async function main() {
  // Create output directories
  await ensureDir(path.join(outDir, 'prs'));
  await ensureDir(path.join(outDir, 'diffs'));

  // Fetch PRs
  await fetchAllPRs(owner, repo, limit);
  log('INFO', `Fetched ${state.totalPRs} PRs`);

  // Fetch missing PR data
  let bar = new cliProgress.SingleBar({
    format: 'Fetching PR details |{bar}| {value}/{total} PRs',
    hideCursor: true
  }, cliProgress.Presets.shades_classic);
  bar.start(state.prsPending.length, 0);
  while (state.prsPending.length > 0) {
    const pr = state.prsPending.pop();
    await fetchMissingDetails(pr);
    state.prsCompleted.push(pr);
    saveState();
    bar.increment();
  }
  bar.stop();

  // Fetch commit diffs
  bar = new cliProgress.SingleBar({
    format: 'Fetching commit diffs |{bar}| {value}/{total} PRs',
    hideCursor: true
  }, cliProgress.Presets.shades_classic);
  bar.start(state.prsCompleted.length, 0);
  while (state.prsCompleted.length > 0) {
    const pr = state.prsCompleted.pop();
    const prFile = path.join(outDir, 'prs', `pr-${pr}.json`);
    const prData = JSON.parse(await fs.readFile(prFile, 'utf-8'));
    const commits = prData.commits?.nodes || [];

    for (const entry of commits) {
      const sha = entry.commit.oid;
      const diffFile = path.join(outDir, 'diffs', `${sha}.diff`);
      if (fsSync.existsSync(diffFile)) continue;

      try {
        const data = await fetchCommitDiff(owner, repo, sha);
        await fs.writeFile(diffFile, data);
      } catch (err) {
        log('ERROR', `Failed to fetch diff for ${sha}: ${err.message}`);
      }
    }

    saveState();
    bar.increment();
  }

  bar.stop();

  log('INFO', 'All phases complete.');
}

main();
