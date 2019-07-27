---
title:  "Introduction to Information Retrieval"
layout: post
categories: Document-Analysis
tags:  Information-Retrieval
author: Suikei Wong
mathjax: true
excerpt_separator: <!--more-->
---

* content
{:toc}

# Information Retrieval

<br>

* Cases: *web search*, *email search*, *image search* ...
* Why: *information overload*
<br><br>

**Basic assumptions.**<br>
**Collection:** a set of *documents*<br>
**Goal:** retrieve documents with information that relevant to the user's *information need* and helps the user complete a task.<br>
**Key objects:** Effectiveness & Accuracy<br>
**IR:** Different to database systems, IR is *unstructured data*.
<br><br><br>

# Term-Document Matrix

<br>

**Search with Boolean query.** E.g., "Obama" AND "healthcare" NOT "news". It's procedures are following:<br>

* Lookup query term in the dictionary
* Retrieve the posting lists
* Operation: AND, OR, NOT

Boolean model provides **all** the ranking candidates (return all the document): <br>
Locate document satisfying **Boolean condition**.<br>
- E.g., "Obama healthcare" -> "Obama" OR "healthcare"<br>
Rank candidates by **relevance**<br>
Efficiency consideration: google uses Top-k retrieval<br><br>
<!--more-->
**Term-document incidence matrices.** Consist of 0/1 and terms. Each row of the matrices is **incidence vectors**. We have a 0/1 vector for each term. To answer the question, we get the incidence vectors that contain all term in the query, then perform the binary operation.
<br><br><br>

# Inverted Index

<br>
**Inverted index** consists of a *dictionary* and *postings*. <br>
![inverted_index](/assets/images/inverted_index.png)<br>
**Dictionary** is a set of unique terms, for each term *t*, we must store a list of all documents that contain *t*: identify each doc by a **docID**.<br>
**Posting** is a variable-size array that keeps the list of documents given term (document id). <br>
**INDEXER** construct inverted index from faw text.<br>
<br>
**Initial stages of text processing:**<br>
- Tokenization: Scan documents for indexable terms and keep list of (token, docID) pairs: <br>
![inverted_index](/assets/images/tokenizer.png)<br>
- Sort tuples by tems and then docID<br>
![inverted_index](/assets/images/sort.png)<br>
- Multiple term entries in a single document are merged, then split into **Dictionary** and **Postings**, so the document frequency information is also added.<br>
![inverted_index](/assets/images/split.png)<br>
<br><br>

# Boolean Retrieval with Inverted Index

<br>

**Linear time retrieval algorithm:** Easy to retrieve all documents containing term *t*, find the same docID from two terms:<br>
![inverted_index](/assets/images/retrieval_alg.png)<br>
**Boolean Retrieval.** Answer any query which is a Boolean expression: AND, OR ,NOT and extended Boolean allows more complex queries.
<br><br><br>

# Document Tokenization

<br>

Assumed that we can easily scan terms from a document, but the scanning consists of following steps:<br>
**Tokenization.** Task of chopping a document into *tokens* by *whitespace* and throwing punctuation are not enough:<br>
E.g., New York is A word; phone numbers and dates cannot be split by *whitespace*; LOWERCASE and LOWER CASE have different meanings, etc<br>
**Regex tokenizer** is simple and efficient and remember always do **the same** tokenization of document and query text.<br>

**Stopwords Removal & Normalization.** Stop words: the most common words in a language: the, a, an, and, etc. Stopwords removal reduce the number of postings that a system has to store since these words are useless.<br>
The normalization of it is to keep **equivalence class** of terms, e.g., U.S.A = USA = united states, and synonym list, e.g., car = automobile, and capitalization and case-folding.<br>

**Stemming and Lemmatization.** Stemming turns tokens into stems, which are the same regardless of inflection (**need not be real words**). Lemmatization turns words into lemmas, which are dictionary entries (**need to be real entries**).