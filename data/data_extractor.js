// fetch-pr-snapshots-with-logging.js
import dotenv from 'dotenv';
import { graphql } from '@octokit/graphql';
import { Octokit } from '@octokit/rest';
import fs from 'fs/promises';
import path from 'path';
import pLimit from 'p-limit';
import pRetry from 'p-retry';

dotenv.config();
const TOKEN = process.env.GITHUB_TOKEN;
if (!TOKEN) throw new Error('Missing GITHUB_TOKEN');

const octokit = new Octokit({ auth: TOKEN });
const gh = graphql.defaults({ headers: { authorization: `token ${TOKEN}` } });

const OWNER    = 'dotnet';
const REPO     = 'runtimelab';
const PER_PAGE = 100;
const SNAP_DIR = path.resolve(process.cwd(), 'snapshots');
const CONCURRENCY = 5;       // max parallel PRs

function log(level, msg) {
  console.log(`[${new Date().toISOString()}] [${level}] ${msg}`);
}

async function ensureSnapshotDir() {
  log('INFO', `Ensuring snapshot directory exists at ${SNAP_DIR}`);
  await fs.mkdir(SNAP_DIR, { recursive: true });
}

// read last processed PRs from disk to resume
async function loadProcessed() {
  try {
    const files = await fs.readdir(SNAP_DIR);
    return new Set(
      files
        .filter(f => f.startsWith('pr-') && f.endsWith('.json'))
        .map(f => parseInt(f.split('-')[1], 10))
    );
  } catch {
    return new Set();
  }
}

async function writeJSON(filePath, data) {
  await fs.writeFile(filePath, JSON.stringify(data, null, 2));
}

// generic retry wrapper for GraphQL and REST calls
async function withRetry(fn, opts = {}) {
  return pRetry(fn, {
    retries: 5,
    factor: 2,
    minTimeout: 1_000,
    maxTimeout: 30_000,
    onFailedAttempt: err => {
      log('WARN', `Attempt ${err.attemptNumber} failed. ${err.retriesLeft} left. ${err.message}`);
    },
    ...opts
  });
}

async function fetchAllPRMetadata() {
    log('INFO', 'Starting fetchAllPRMetadata()');
    let cursor = null;
    const prs = [];
  
    while (true) {
      // fetch a page of PRs
      const resp = await withRetry(() =>
        gh(
          `query($owner:String!,$repo:String!,$first:Int!,$after:String) {
             repository(owner:$owner, name:$repo) {
               pullRequests(first:$first, after:$after) {
                 pageInfo { hasNextPage endCursor }
                 nodes {
                   number
                   title
                   body
                   createdAt
                   labels(first:10) { 
                     nodes { name } 
                   }
                 }
               }
             }
           }`,
          { owner: OWNER, repo: REPO, first: PER_PAGE, after: cursor }
        )
      );
  
      const conn = resp.repository.pullRequests;
      // map each node into a plain object
      conn.nodes.forEach(node => {
        prs.push({
          number:    node.number,
          title:     node.title,
          body:      node.body,
          createdAt: node.createdAt,
          labels:    node.labels.nodes.map(l => l.name)
        });
      });
  
      log('DEBUG', `Fetched ${conn.nodes.length} PRs, total so far: ${prs.length}`);
  
      if (!conn.pageInfo.hasNextPage) break;
      cursor = conn.pageInfo.endCursor;
    }
  
    // write out a snapshot of the full PR metadata
    const snap = path.join(
      SNAP_DIR,
      `pr-metadata-${new Date().toISOString()}.json`
    );
    await writeJSON(snap, prs);
    log('INFO', `PR metadata snapshot → ${snap}`);
  
    return prs;
}

async function fetchConnection(prNumber, fieldName, selection) {
  let cursor = null;
  const items = [];

  while (true) {
    const resp = await withRetry(() =>
      gh(
        `query($owner:String!,$repo:String!,$pr:Int!,$first:Int!,$after:String){
           repository(owner:$owner, name:$repo){
             pullRequest(number:$pr){
               ${fieldName}(first:$first, after:$after){
                 pageInfo { hasNextPage endCursor }
                 nodes { ${selection} }
               }
             }
           }
         }`,
        { owner: OWNER, repo: REPO, pr: prNumber, first: PER_PAGE, after: cursor }
      )
    );
    const conn = resp.repository.pullRequest[fieldName];
    items.push(...conn.nodes);
    if (!conn.pageInfo.hasNextPage) break;
    cursor = conn.pageInfo.endCursor;
  }
  return items;
}

async function fetchReviewComments(prNumber) {
  let cursor = null;
  const comments = [];

  while (true) {
    const resp = await withRetry(() =>
      gh(
        `query($owner:String!,$repo:String!,$pr:Int!,$first:Int!,$after:String){
           repository(owner:$owner, name:$repo){
             pullRequest(number:$pr){
               reviewThreads(first:$first, after:$after){
                 pageInfo { hasNextPage endCursor }
                 nodes {
                   comments(first:${PER_PAGE}) {
                     nodes {
                       path diffHunk body createdAt author { login }
                     }
                   }
                 }
               }
             }
           }
         }`,
        { owner: OWNER, repo: REPO, pr: prNumber, first: PER_PAGE, after: cursor }
      )
    );

    const threads = resp.repository.pullRequest.reviewThreads.nodes;
    threads.forEach(t => t.comments.nodes.forEach(c => comments.push(c)));
    const pageInfo = resp.repository.pullRequest.reviewThreads.pageInfo;
    if (!pageInfo.hasNextPage) break;
    cursor = pageInfo.endCursor;
  }
  return comments;
}

async function enrichAndSnapshotPR(prNumber) {
  log('INFO', `→ Enriching PR #${prNumber}`);
  // parallel fetch of comments / commits meta / review-comments
  const [prComments, commitsMeta, prReviewComments] = await Promise.all([
    fetchConnection(prNumber, 'comments', `id body createdAt author { login }`),
    fetchConnection(prNumber, 'commits',  `commit { oid message committedDate }`),
    fetchReviewComments(prNumber)
  ]);

  // now fetch full diffs in parallel (limited)
  const limit = pLimit(CONCURRENCY);
  const commits = await Promise.all(
    commitsMeta.map(({ commit }) =>
      limit(async () => {
        log('DEBUG', ` Fetching diff for ${commit.oid}`);
        const { data: diff } = await withRetry(() =>
          octokit.request('GET /repos/{owner}/{repo}/commits/{ref}', {
            owner: OWNER,
            repo: REPO,
            ref: commit.oid,
            mediaType: { format: 'diff' }
          })
        );
        return { ...commit, diff };
      })
    )
  );

  const raw = { prNumber, comments: prComments, reviewComments: prReviewComments, commits };
  const filename = `pr-${prNumber}-${new Date().toISOString()}.json`;
  const filepath = path.join(SNAP_DIR, filename);
  await writeJSON(filepath, raw);
  log('INFO', ` PR#${prNumber} snapshot → ${filepath}`);
  return raw;
}

(async () => {
  try {
    await ensureSnapshotDir();
    const processed = await loadProcessed();
    const prs = await fetchAllPRMetadata();

    const limit = pLimit(CONCURRENCY);
    const enriched = [];
    for (const pr of prs) {
      if (processed.has(pr)) {
        log('INFO', `Skipping PR#${pr} (already done)`);
        continue;
      }
      enriched.push(limit(() => enrichAndSnapshotPR(pr.number)));
    }
    await Promise.all(enriched);

    await writeJSON(allFile, allData);
    log('INFO', `All-PRs snapshot → ${allFile}`);
    log('INFO', 'Done.');
  } catch (err) {
    log('ERROR', `Fatal error: ${err.message}`);
    process.exit(1);
  }
})();
