---
title:  "Information Retrieval"
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

**Term-document incidence matrices.** Consist of 0/1 and terms. Each row of the matrices is **incidence vectors**. We have a 0/1 vector for each term. To answer the question, we get the incidence vectors that contain all term in the query, then perform the binary operation.<br>
- E.g., **Brutus**, **Caesar** and **Calpurnia** -> 110110 AND 110111 AND 010000 = 010000
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
- E.g., New York is A word; phone numbers and dates cannot be split by *whitespace*; LOWERCASE and LOWER CASE have different meanings, etc<br>

**Regex tokenizer** is simple and efficient and remember always do **the same** tokenization of document and query text.<br>

**Stopwords Removal & Normalization.** Stop words: the most common words in a language: the, a, an, and, etc. Stopwords removal reduce the number of postings that a system has to store since these words are useless.<br>
The normalization of it is to keep **equivalence class** of terms, e.g., U.S.A = USA = united states, and synonym list, e.g., car = automobile, and capitalization and case-folding.<br>

**Stemming and Lemmatization.** Stemming turns tokens into stems, which are the same regardless of inflection (**need not be real words**). Lemmatization turns words into lemmas, which are dictionary entries (**need to be real entries**).

# Boolean Retrieval

<br>

**Bag-of-Words (BOW) assumption.** We didn't care about the ordring of tokens in documents. <br>
- E.g., Mary married John == John married Mary

**Field in document.** Document is a semi-structured data: title, author, published date, body, etc... <br>
**Basic Field Index.** term.field: Fields are encoded as extension of dictionary entries: <br>
![basic_field_index](/assets/images/basic_index.png)<br>
**Field in Posting.** Convert the fields in the posting list instead of the dictionary: <br>
![field_in_posting](/assets/images/field_posting.png)<br>
<br><br><br>

# Ranked Retrieval

<br>

*Given a query, rank documents according to some criterion so that the "best" results appear early in the result list displayed to the user.*<br>
The goal if ranked retrieval is to find a scoring function:<br>
<center>$$ Score(d,q) $$</center>
where $$ d $$ is a document $$ q $$ is a query.<br>
<br>
**Weighted fields approach.** In this approach, importance of term is not the same, and we assign different weights to terms based on their fields.<br>
**Scoring with weighted fields.** The sum of the weights of all fields is 1, and we calculate the score of terms in document through this:<br>
<center>$$\text {Score}(d, t)=\sum_{i=1}^{\ell} g_{i} \times s_{i} \quad \text { where }\left\{\begin{array}{l}{s_{i}=1, \text { if } t \text { is in field } i \text { of } d} \\ {s_{i}=0, \text { otherwise }}\end{array}\right.$$</center>
where $$ t $$ is a query term, $$ d $$ is the document, $$ \ell $$ is the number of fields and $$ i $$ is the ith fields, $$ g_{i} $$ is the weight of field $$ i $$.<br>
Notice that a score of a query term ranges [0, 1].<br>
<br>
**Rank by term frequency.** **Term Frequency (TF)** $$ tf_{t,d} $$ is the number of occurences of term $$ t $$ in document $$ d $$. **It's the number of term**. Let $$ q $$ be a set of query terms $$ \left(t_{1}, t_{2}, \dots, t_{m}\right) $$, a **term frequency score** of documents given query $$ q $$ is:<br>
<center>$$ \text {Score}_{t f}(d, q)=\sum_{i=1}^{m} \mathrm{tf}_{t_{i}, d} $$</center><br>
In this case, the score is defined by the frequency only, no weights or fields. **The numbers of a term appeared in a document higher, this rank of this document is higher.** <br>
<br>
**Document frequency.** **Document Frequency(DF)** $$ df_{t} $$ is the number of documents in the collection that contain term $$ t $$. **It's the number of documents**.<br>
**df** is a good way to measure an importance of a term. If a word appears in many documents, it's not important. **Less frequency, more unique, more important.**
- High frequency -> not important (e.g., stopwords)
- Low frequency -> important 

**Innverse document frequency.** **IDF** $$ df_{t} $$ is the number of documents in the collection that contain a term $$ t $$. The inverse document frequency (IDF) can be defined as follows:<br>
<center>$$ idf_{t}=\log \left(\frac{N}{d f_{t}}\right) $$</center>
where $$ N $$ is the total number of documents. <br>
The idf of a rare term is high, which is more important, whereas the idf of a frequent term is likely to be low, which is less important.
<br><br>
**TF-IDF** The tf-idf weight of term $$ t $$ in document $$ d $$ is as follows:<br>
<center>$$ \mathrm{tf}-\mathrm{idf}_{t, d}=\mathrm{tf}_{t, d} \times \mathrm{idf}_{t} $$</center><br>
With tf-idf weighting scheme, the score of document $$ d $$ given query $$ q $$ is:<br>
<center>$$ Score_{tr-idf}(d, q)=\sum_{i=1}^{m} \mathrm{tf}-\mathrm{idf}_{t_{i}, d}$$</center><br>
<br>
**Variants of tf-idf.** Since tf-idf scoring still heavily relies on the frequency of terms (score linearly increases with respect to frequency of term), after a certain frequency, the absolute frequency isn't important. So we use other developed scaling.<br>
**Sublinear tf scaling.** Use logarithmically weighted term frequence (wf):<br>
<center>$$ \mathrm{wf}_{t, d}=\left\{\begin{array}{ll}{1+\log \mathrm{tf}_{t, d},} & {\text { if } \mathrm{tf}_{t, d}>0} \\ {0} & {\text { otherwise }}\end{array}\right. $$</center><br>
In this case, the logarithmic term frequency version of **tf-idf** is: <br>
<center>$$ \mathrm{wf}-\mathrm{idf}_{t, d}=\mathrm{wf}_{t, d} \times \mathrm{idf}_{f} $$</center><br>
**Maximum tf normalization.** tf-idf and wf-idf scoring still has limitation: if we only appending a copy of document, the term frequency is increased, but the information doesn't changed, since scoring prefers longer documents. In **maximum tf normalization**, $$ tf_{max}(d) $$ is the maximum frequency of document $$ d $$, normalized term frequency is defined as:<br>
<center>$$ \operatorname{ntf}_{t, d}=\alpha+(1-\alpha) \frac{\mathrm{tf}_{t, d}}{\mathrm{tf}_{\max }(d)} \quad \text { if } \mathrm{tf}_{t, d}>0 $$</center><br>
$$ \alpha = 0.4/0.5 $$
- Maximum value of ntf is 1:$$ tf_{t,d} = tf_{max}(d) $$
- Minimum value of ntf is $$ \alpha $$: $$ tf=0 $$

<br><br><br>
# Relevance Feedback

<br>

**Document as Vectors.** Given a term-document matrix, every document can be represented as a vector or length $$ V $$, which is the size of vocabulary.<br>
**Document Similarity in Vector Space.** Plot document vectors in vector space, find the similar documents in vector space:
- Distance from vector to vector
- Angle difference between vectors

**Angel Difference.** Consine similarity:<br>
<center>$$ \operatorname{sim}\left(\vec{d}_{1}, \vec{d}_{2}\right)=\frac{\vec{d}_{1} \cdot \vec{d}_{2}}{\left|\vec{d}_{1}\right| \times\left|\vec{d}_{2}\right|} $$</center><br>
Numerator is the inner product, denominator is the product of Euclidean lengths.<br>
Standard way of quantifying similarity:
- If directions of two vectors are the same, the result is **1**.
- If directions of two vector are orthogonal(90), the result is **0**.

**Query as Document.** Query can be converted as vector too. We can also compute the similarity between query and document. **Score function for vector space model:** <br>
<center>$$ \mathrm{Score}_{\mathrm{vsm}}(d, q)=\operatorname{sim}(\vec{d}, \vec{q}) $$</center><br>
<br>
Actually relevance feedback is asking explicit feedback from user, instead of letting users refine query. The system needs to recompute user's information need based on their feedback and displays a revised set of results.<br>
**Relevance feedback in vector space.** Documents just like some dot in the space, and the initial query is also in this space. Find the relevant and irrelevant documents similar to the initial query, then revise initial query internally to return a new set of documents.
<br><br><br>
# Tolerant Retrieval

<br>

Recall the dictionary data structures. It stores the term vocabulary, document frequency and pointers to each posting list. There are two main choices of structures, **Hashtables and Trees.**<br>
**Hashtables.** Each term is harshed to an integer, this way's lookup is faster than for a tree, but it's no easy way to find minor variants and no prefix search. If vocabulary keeps growing, need to occasionally do the expensive operation of rehashing everything.<br>
**Tree.** Simplest: Binary tree. More usual: B-tree. Trees require a standard ordering of characters and hence strings. It can solves the prefix problem, but it's slower and reblancing it is expensive.<br><br>

**Wild-card queries.** Use `*` as the wild-card queries to find words. E.g., `mon*` find all docs containing any word begining with `mon`. If we need to find words ending in `mon`, we need to maintain an additional B-tree for terms *backwards*. <br>
**Query processing.** All terms in the dictionary that match the wild-card query and we still look up the postings for each enumerated term.<br>
**Permuterm index.** If we need to handle `*'s` in the middle of query term, it's expensive to look up two sets in a B-tree and intersect the two term sets. **Permuterm index** transform wild-card queries:
- E.g., for the term *hello*, index under: `hello$, ello$h, llo$he, lo$hel, o$hell, $hello`, where `$` is a special symbol.

In **permuterm query processing**, it rotates query wild-card to the right, and uses B-tree lookup as before.<br><br>
**Bigram (*k-gram*) indexes.** Enumerate all *k-grams* occurring in any term. Maintain a *second* inverted index *from bigrams to dictionary terms* that match each bigram. To processing wild-cards, e.g., query `mon*` can now be run as `$m AND mo AND on`. But it will enumerate some error, e.g., `moon`. So we must ***post-filter*** these terms against query. <br><br>
**Spelling correction.** Correct documents being indexed and user queries to retrieve "right" answers.
- Isolated word: check each word on its own for misspelling.
- Context-sensitive: look at surrounding words.

**Document correction.** Especially needed for OCR‘ed documents. But also in web pages and even printed material have typos. The goal of document correction is to make the dictionary contains fewer misspellings, but often we don't change the documents and instead **fix the query-document mapping**.<br>
**Query mis-spellings.** We can retrieve documents indexed by the correct spelling, or return several suggested alternative queries with the correct spelling: e.g., *do you mean ...?*<br>
**Isolated word correction.** *Fundamental premises:* there is a lexicon from which the correct spelling come: a standard lexicon and the lexicon of the indexed corpus (e.g., all words on the web). Given a lexicon and a character sequences Q, return the words in the lexicon closest to Q. How we define the closest? <br><br>
**Edit distance.** Counting the number of operations that you convert one to the other: Given two strings $$ S_{1} $$ and $$ S_{2} $$. Operations are typically character-level like insert, delete, replace and transposition. <br>
**Weighted edit distance.** Added **weight** to an operation. It's depends on the characters involved. E.g., replacing ***m*** by ***n*** is small edit distance since ***m*** is more likely to be mis-typed as ***n***. This method requires weight matrix as input and modify dynamic programming to handle weights. <br>
**Using edit distances.** Given query:
- first **enumerate all character sequences** with a preset(weighted) edit distance
- intersect this set with list of **"correct"** words
- show terms you found to user as suggestions

<br>
**n-gram overlap.** We cannot compute its edit distance to every dictionary term since it's expensive and slow. We **cut** the set of candidate dictionary terms by using **n-gram overlap**
- enumerate all the **n-grams** in the query string and lexicon
- use **n-gram index**(wild-card search) to retrieve all lexicon terms matching any of the query n-grams
- **threshold by number of matching n-grams** (e.g., weight by keyboard layout)


