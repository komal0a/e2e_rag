import * as dotenv from 'dotenv';
dotenv.config();

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { Pinecone } from '@pinecone-database/pinecone';
import { v4 as uuid } from 'uuid';
import fs from 'fs';

const API_KEY = process.env.GEMINI_API_KEY;

// ─── Embedding function ───────────────────────────────────────────────
async function getEmbedding(text, retries = 10) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=${API_KEY}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'models/gemini-embedding-001',
          content: { parts: [{ text }] },
        }),
      }
    );
    const data = await response.json();

    if (data.embedding?.values) {
      return data.embedding.values.slice(0, 768);
    }

    if (data.error?.status === 'RESOURCE_EXHAUSTED') {
      console.log(`  Rate limit hit — 70s wait... (attempt ${attempt}/${retries})`);
      await new Promise(r => setTimeout(r, 70000));
      continue;
    }

    console.error('Embedding error:', JSON.stringify(data));
    throw new Error('Embedding failed');
  }
  throw new Error('Max retries exceeded');
}

// ─── Cosine Similarity ───────────────────────────────────────────────
function cosineSimilarity(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot  += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

// ─── Sentence Splitter ───────────────────────────────────────────────
function splitIntoSentences(text) {
  return text
    .replace(/\n+/g, ' ')
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(s => s.length > 20);
}

// ─── Semantic Chunking (PRODUCTION version) ──────────────────────────
// Yeh production mein use hoti hai — har sentence embed karke
// similarity ke basis par chunks banata hai
// Free tier pe expensive hai (938 API calls) isliye neeche
// smartChunk use kar rahe hain
async function semanticChunk(text, threshold = 0.95) {
  const sentences = splitIntoSentences(text);
  if (sentences.length === 0) return [];

  console.log(`  ${sentences.length} sentences mile, embed ho rahe hain...`);

  const vectors = [];
  for (let i = 0; i < sentences.length; i++) {
    const vector = await getEmbedding(sentences[i]);
    vectors.push(vector);
    await new Promise(r => setTimeout(r, 1200));
    if ((i + 1) % 50 === 0) console.log(`  Embedded ${i + 1}/${sentences.length}`);
  }
  console.log(`  Embedded ${sentences.length}/${sentences.length}`);

  const chunks = [];
  let currentGroup = [sentences[0]];

  for (let i = 1; i < sentences.length; i++) {
    const similarity  = cosineSimilarity(vectors[i - 1], vectors[i]);
    const forcedBreak = currentGroup.length >= 15;
    if (similarity < threshold || forcedBreak) {
      chunks.push(currentGroup.join(' '));
      currentGroup = [sentences[i]];
    } else {
      currentGroup.push(sentences[i]);
    }
  }
  if (currentGroup.length > 0) chunks.push(currentGroup.join(' '));

  console.log(`  ${chunks.length} chunks bane`);
  return chunks;
}

// ─── Smart Chunking (FREE TIER version) ─────────────────────────────
// Yeh free tier ke liye hai — zero extra API calls
// Sirf sentences ko fixed groups mein divide karta hai
// Interview mein batao: "Production mein semanticChunk use hoti,
// free tier quota ki wajah se smartChunk use kar rahe hain"
function smartChunk(text, maxSentences = 8) {
  const sentences = splitIntoSentences(text);
  const chunks = [];
  for (let i = 0; i < sentences.length; i += maxSentences) {
    const group = sentences.slice(i, i + maxSentences);
    chunks.push(group.join(' '));
  }
  console.log(`  ${chunks.length} chunks bane`);
  return chunks;
}

// ─── BM25 ────────────────────────────────────────────────────────────
class SimpleBM25 {
  constructor() {
    this.documents = [];
    this.k1 = 1.5;
    this.b  = 0.75;
  }

  tokenize(text) {
    return text.toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .split(/\s+/)
      .filter(Boolean);
  }

  addDocument(id, text) {
    const tokens = this.tokenize(text);
    this.documents.push({ id, text, tokens });
  }

  save(path) {
    fs.writeFileSync(path, JSON.stringify(this.documents, null, 2));
  }
}

// ─── Main ────────────────────────────────────────────────────────────
async function indexDocument() {
  const PDF_PATH = './dsa.pdf';

  // Step 1: PDF load karo
  const pdfLoader = new PDFLoader(PDF_PATH);
  const rawDocs   = await pdfLoader.load();
  console.log('PDF loaded');

  const fullText = rawDocs.map(d => d.pageContent).join('\n');
  console.log('Embedding model configured');

  // Step 2: Chunking
  // NOTE: Production mein semanticChunk use karo
  // Free tier quota ke liye smartChunk use kar rahe hain
  console.log('Smart chunking shuru...');
  const chunkTexts  = smartChunk(fullText); // swap to semanticChunk() for production
  const cleanChunks = chunkTexts.filter(t => t.trim().length > 50);
  console.log(`Chunking complete — ${cleanChunks.length} clean chunks bane`);

  // Step 3: Har chunk ko unique ID do
  const chunkIds = cleanChunks.map(() => uuid());

  // Step 4: BM25 index banao aur save karo
  const bm25 = new SimpleBM25();
  cleanChunks.forEach((text, i) => bm25.addDocument(chunkIds[i], text));
  bm25.save('./bm25_index.json');
  console.log('BM25 index saved to bm25_index.json');

  // Step 5: Pinecone configure karo
  const pinecone      = new Pinecone();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
  console.log('Pinecone configured');

  // Step 6: Chunks embed karo — 1.2s delay rate limit avoid karta hai
  console.log(`Final embeddings bana rahe hain — ${cleanChunks.length} chunks...`);
  const upsertData = [];

  for (let i = 0; i < cleanChunks.length; i++) {
    const vector = await getEmbedding(cleanChunks[i]);
    upsertData.push({
      id:     chunkIds[i],
      values: vector,
      metadata: {
        text:       cleanChunks[i],
        source:     PDF_PATH,
        chunkIndex: i,
      },
    });

    await new Promise(r => setTimeout(r, 1200));

    if ((i + 1) % 10 === 0) {
      console.log(`  Embedded ${i + 1}/${cleanChunks.length} chunks`);
    }
  }

  console.log(`All embeddings done — dimension: ${upsertData[0].values.length}`);

  // Step 7: Pinecone mein 100 at a time upsert karo
  const BATCH = 100;
  for (let i = 0; i < upsertData.length; i += BATCH) {
    await pineconeIndex.upsert(upsertData.slice(i, i + BATCH));
    console.log(`Pinecone batch ${Math.floor(i / BATCH) + 1} stored`);
  }

  console.log(`✓ Data stored in Pinecone successfully!`);
  console.log(`✓ Total chunks indexed: ${upsertData.length}`);
}

indexDocument();