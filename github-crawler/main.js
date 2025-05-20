import dotenv from 'dotenv';
import { Octokit } from '@octokit/rest';
import { graphql } from '@octokit/graphql';
import fs from 'fs/promises';
import fsSync from 'fs';
import path from 'path';
import pLimit from 'p-limit';
import pRetry from 'p-retry';
import { promisify } from 'util';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

// Load environment
dotenv.config();
const TOKEN = process.env.GITHUB_TOKEN;
if (!TOKEN) throw new Error('Missing GITHUB_TOKEN');

// CLI configuration
const argv = yargs(hideBin(process.argv))
  .option('owner',       { type: 'string', demandOption: true, describe: 'GitHub repo owner' })
  .option('repo',        { type: 'string', demandOption: true, describe: 'GitHub repo name' })
  .option('outDir',      { type: 'string', default: 'output', describe: 'Base output directory' })
  .option('concurrency', { type: 'number', default: 5, describe: 'Max parallel requests' })
  .option('maxPrs',      { type: 'number', describe: 'Limit to first N PRs for testing' })
  .option('skipDiff',    { type: 'boolean', default: false, describe: 'Skip fetching diffs' })
  .argv;

const { owner, repo, outDir, concurrency, maxPrs, skipDiff } = argv;
const octokit = new Octokit({ auth: TOKEN });
const gh = graphql.defaults({ headers: { authorization: `token ${TOKEN}` } });
const sleep = promisify(setTimeout);

// Logging helper
function log(level, msg) {
  console.log(`[${new Date().toISOString()}] [${level}] ${msg}`);
}

// Rate-limit handling
async function waitForRateLimitReset() {
  const { data } = await octokit.rest.rateLimit.get();
  const reset = data.resources.core.reset * 1000;
  const ms = Math.max(reset - Date.now(), 0);
  log('WARN', `Rate limit hit. Sleeping ${Math.ceil(ms/1000)}s until ${new Date(reset).toISOString()}`);
  await sleep(ms + 1000);
}

// Proactive rate limit check
async function checkRateLimit(threshold = 10) {
  const { data } = await octokit.rest.rateLimit.get();
  const remaining = data.resources.core.remaining;
  const reset = data.resources.core.reset * 1000;
  if (remaining < threshold) {
    const ms = Math.max(reset - Date.now(), 0);
    log('WARN', `Approaching rate limit (${remaining} left). Sleeping ${Math.ceil(ms/1000)}s until ${new Date(reset).toISOString()}`);
    await sleep(ms + 1000);
  }
}

// Smarter retry wrapper: only retry on network/5xx or rate limit errors
async function withSmartRetry(fn) {
  return pRetry(fn, {
    retries: 5,
    factor: 2,
    minTimeout: 1000,
    maxTimeout: 30000,
    onFailedAttempt: err => log('WARN', `Attempt #${err.attemptNumber} failed: ${err.message}`),
    retry: err => {
      if (/rate limit/i.test(err.message)) return true;
      if (err.status && err.status >= 500) return true;
      if (err.code === 'ENOTFOUND' || err.code === 'ECONNRESET') return true;
      return false;
    }
  });
}

// Ensure directory exists
async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true });
}

// GraphQL helper with rate-limit handling
async function callGh(query, variables) {
  try {
    return await gh(query, variables);
  } catch (err) {
    if (/rate limit/i.test(err.message)) {
      await waitForRateLimitReset();
      return gh(query, variables);
    }
    throw err;
  }
}

// Fetch a paginated connection from GraphQL
async function fetchConnection(prNumber, field, selection) {
  let cursor = null;
  const items = [];
  do {
    const resp = await withSmartRetry(() => callGh(
      `query($owner:String!,$repo:String!,$pr:Int!,$first:Int!,$after:String){
         repository(owner:$owner, name:$repo) {
           pullRequest(number:$pr) {
             ${field}(first:$first, after:$after) {
               pageInfo { hasNextPage endCursor }
               nodes { ${selection} }
             }
           }
         }
       }`,
      { owner, repo, pr: prNumber, first: 100, after: cursor }
    ));
    const conn = resp.repository.pullRequest[field];
    items.push(...conn.nodes);
    cursor = conn.pageInfo.endCursor;
  } while (cursor);
  return items;
}

// Fetch list of PRs with metadata
async function fetchAllPRs() {
  let cursor = null;
  const prs = [];

  do {
    const resp = await withSmartRetry(() => callGh(
      `query($owner:String!,$repo:String!,$first:Int!,$after:String){
         repository(owner:$owner, name:$repo) {
           pullRequests(first:$first, after:$after) {
             pageInfo { hasNextPage endCursor }
             nodes {
               number
               title
               body
               createdAt
               labels(first:10){ nodes { name } }
             }
           }
         }
       }`,
      { owner, repo, first: 100, after: cursor }
    ));

    const conn = resp.repository.pullRequests;
    prs.push(...conn.nodes.map(n => ({
      number:    n.number,
      title:     n.title,
      body:      n.body,
      createdAt: n.createdAt,
      labels:    n.labels.nodes.map(l => l.name)
    })));

    cursor = conn.pageInfo.endCursor;
  } while (cursor && prs.length < (maxPrs || Infinity));

  return maxPrs ? prs.slice(0, maxPrs) : prs;
}

// Fetch diff for a commit SHA
async function fetchDiff(sha) {
  const dir = path.join(outDir, 'diffs');
  const file = path.join(dir, `${sha}.diff`);
  if (fsSync.existsSync(file)) return;

  await checkRateLimit();
  await withSmartRetry(async () => {
    try {
      const res = await octokit.request(
        'GET /repos/{owner}/{repo}/commits/{sha}',
        { owner, repo, sha, headers: { accept: 'application/vnd.github.v3.diff' } }
      );
      await ensureDir(dir);
      await fs.writeFile(file, res.data);
      log('INFO', `Saved diff ${sha}`);
    } catch (err) {
      if (/rate limit/i.test(err.message)) {
        await waitForRateLimitReset();
        throw err;
      }
      if (err.status >= 400 && err.status < 500) {
        log('WARN', `Skipping diff ${sha}: Error code ${err.status}`);
        return;
      }
      throw err;
    }
  });
}

// Phase 1: Fetch and save all PR metadata
async function fetchAndSaveAllPRs() {
  await ensureDir(path.join(outDir, 'prs'));
  const prs = await fetchAllPRs();
  log('INFO', `Found ${prs.length} PRs`);
  for (const pr of prs) {
    try {
      await checkRateLimit();
      const [comments, reviewThreads, commitsMeta] = await Promise.all([
        fetchConnection(pr.number, 'comments', 'id body createdAt author{login}'),
        fetchConnection(pr.number, 'reviewThreads', 'comments(first:100){nodes{path diffHunk body createdAt author{login}}}'),
        fetchConnection(pr.number, 'commits', 'commit{oid message committedDate}')
      ]);
      const reviewComments = reviewThreads.flatMap(t => t.comments.nodes);
      const data = {
        prNumber:   pr.number,
        title:      pr.title,
        body:       pr.body,
        createdAt:  pr.createdAt,
        labels:     pr.labels,
        comments,
        reviewComments,
        commitsMeta
      };
      const file = path.join(outDir, 'prs', `pr-${pr.number}.json`);
      await fs.writeFile(file, JSON.stringify(data, null, 2));
      log('INFO', `Saved PR#${pr.number}`);
    } catch (err) {
      log('ERROR', `Failed to fetch/save PR#${pr.number}: ${err.message}`);
    }
  }
}

// Phase 2: Fetch all commit diffs for all PRs
async function fetchAllDiffs() {
  const prsDir = path.join(outDir, 'prs');
  const diffDir = path.join(outDir, 'diffs');
  await ensureDir(diffDir);
  const files = fsSync.readdirSync(prsDir).filter(f => f.startsWith('pr-') && f.endsWith('.json'));
  const limit = pLimit(concurrency);
  for (const file of files) {
    let data;
    try {
      data = JSON.parse(fsSync.readFileSync(path.join(prsDir, file), 'utf-8'));
    } catch (err) {
      log('ERROR', `Failed to read/parse ${file}: ${err.message}`);
      continue;
    }
    if (!data.commitsMeta) continue;
    const shas = data.commitsMeta.map(c => c.commit.oid);
    await Promise.all(shas.map(sha => limit(async () => {
      try {
        await fetchDiff(sha);
      } catch (err) {
        log('ERROR', `Failed to fetch diff for ${sha} in ${file}: ${err.message}`);
      }
    })));
  }
}

(async () => {
  try {
    await fetchAndSaveAllPRs();
    await fetchAllDiffs();
    log('INFO', 'All done.');
  } catch (err) {
    log('ERROR', err.message);
    process.exit(1);
  }
})();
