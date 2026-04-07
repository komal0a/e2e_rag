// Chat karne ka file — ab hybrid search ke saath (vector + keyword)
import * as dotenv from 'dotenv';
dotenv.config();

import readlineSync from 'readline-sync';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from '@google/genai';
import fs from 'fs';

const ai = new GoogleGenAI({});
const History = [];

// ─── BM25 Index Load karo ────────────────────────────────────────────
// index.js ne yeh file banayi thi, ab hum ise read karenge

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

  load(path) {
    this.documents = JSON.parse(fs.readFileSync(path, 'utf-8'));
  }

  search(query, topK = 10) {
    const queryTokens = this.tokenize(query);
    const avgLen = this.documents.reduce((s, d) => s + d.tokens.length, 0) / this.documents.length;

    const scored = this.documents.map(doc => {
      let score = 0;
      const tokenCounts = {};
      doc.tokens.forEach(t => { tokenCounts[t] = (tokenCounts[t] || 0) + 1; });

      queryTokens.forEach(qt => {
        const tf = tokenCounts[qt] || 0;
        if (tf === 0) return;
        const idf = Math.log(1 + (this.documents.length - 1) / (1 + this.documents.filter(d => d.tokens.includes(qt)).length));
        const num  = tf * (this.k1 + 1);
        const den  = tf + this.k1 * (1 - this.b + this.b * doc.tokens.length / avgLen);
        score += idf * (num / den);
      });

      return { id: doc.id, text: doc.text, score };
    });

    return scored
      .filter(d => d.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
}

// ─── RRF Fusion ──────────────────────────────────────────────────────
// Vector results aur BM25 results ko ek mein merge karna
// RRF = Reciprocal Rank Fusion — dono lists ki rank combine karo

function rrfFusion(vectorResults, bm25Results, topK = 6) {
  // k=60 ek standard value hai jo extreme ranks ka impact smooth karta hai
  const K = 60;
  const scores = new Map();

  const addScores = (results) => {
    results.forEach((result, rank) => {
      const id  = result.id || result.metadata?.id;
      const text = result.pageContent || result.text;
      const prev = scores.get(id) || { score: 0, text };
      scores.set(id, {
        score: prev.score + 1 / (K + rank + 1),
        text: prev.text || text,
      });
    });
  };

  addScores(vectorResults);
  addScores(bm25Results);

  return [...scores.entries()]
    .sort((a, b) => b[1].score - a[1].score)
    .slice(0, topK)
    .map(([id, { score, text }]) => ({ id, score, text }));
}

// ─── Query Transform (same as your original) ─────────────────────────

async function transformQuery(question) {
  History.push({ role: 'user', parts: [{ text: question }] });

  const response = await ai.models.generateContent({
    model: 'gemini-2.0-flash',
    contents: History,
    config: {
      systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
Only output the rewritten question and nothing else.`,
    },
  });

  History.pop();
  return response.text;
}

// ─── Main Chat Function ───────────────────────────────────────────────

async function chatting(question) {
  // Step 1: Query transform karo (same as your original)
  const rewrittenQuery = await transformQuery(question);
  console.log(`\nRewritten query: ${rewrittenQuery}`);

  // Step 2: Embedding model (same as your original)
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'text-embedding-004',
  });

  // Step 3: Query ko vector mein convert karo (same as your original)
  const queryVector = await embeddings.embedQuery(rewrittenQuery);

  // Step 4: Pinecone se vector search karo (same as your original)
  const pinecone      = new Pinecone();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

  const vectorSearchResult = await pineconeIndex.query({
    topK: 10,
    vector: queryVector,
    includeMetadata: true,
  });

  // Pinecone results ko standard format mein convert karo
  const vectorResults = vectorSearchResult.matches.map(m => ({
    id: m.metadata.id,
    text: m.metadata.text,
    score: m.score,
  }));

  // Step 5: BM25 keyword search karo (NEW)
  const bm25 = new SimpleBM25();
  bm25.load('./bm25_index.json');
  const bm25Results = bm25.search(rewrittenQuery, 10);

  // Step 6: RRF se dono results merge karo (NEW)
  const hybridResults = rrfFusion(vectorResults, bm25Results, 6);
  console.log(`Hybrid search: ${hybridResults.length} chunks mili`);

  // Step 7: Context banao (same structure as your original)
  const context = hybridResults
    .map(r => r.text)
    .join('\n\n---\n\n');

  // Step 8: Gemini se answer lo (same as your original)
  History.push({ role: 'user', parts: [{ text: rewrittenQuery }] });

  const response = await ai.models.generateContent({
    model: 'gemini-2.0-flash',
    contents: History,
    config: {
      systemInstruction: `You have to behave like a Data Structure and Algorithm Expert.
You will be given a context of relevant information and a user question.
Your task is to answer the user's question based ONLY on the provided context.
If the answer is not in the context, you must say "I could not find the answer in the provided document."
Keep your answers clear, concise, and educational.

Context: ${context}`,
    },
  });

  History.push({ role: 'model', parts: [{ text: response.text }] });

  console.log('\n' + response.text);

  // Evaluation ke liye yeh data return karo (evaluate.js use karega)
  return {
    question,
    rewrittenQuery,
    answer: response.text,
    contexts: hybridResults.map(r => r.text),
  };
}

// ─── Main Loop (same as your original) ──────────────────────────────

async function main() {
  const userProblem = readlineSync.question('Ask me anything --> ');
  await chatting(userProblem);
  main();
}

main();