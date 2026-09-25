import math
import re


from nltk.stem import PorterStemmer
from pathlib import Path
from collections import Counter, defaultdict



stemmer = PorterStemmer()

STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on"}

def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stemmer.stem(w) for w in words if w not in STOPWORDS]

class SearchEngine:
    def __init__(self, k1=1.5, b=0.75):
        self.index = defaultdict(dict) # word {doc_id: count}
        self.docs = {}                 # doc_id -> original
        self.doc_len = {}              # doc_id -> number of words
        self.k1 = k1
        self.b = b


    def add(self, doc_id, text):
        tokens = tokenize(text)
        self.docs[doc_id] = text
        self.doc_len[doc_id] = len(tokens)
        for word, count in Counter(tokenize(text)).items():
            self.index[word][doc_id] = count

    def add_folder(self, folder):
        for path in sorted(Path(folder).rglob("*")):
            if path.suffix in {".txt", ".md"}:
                self.add(str(path), path.read_text(encoding="utf-8", errors="ignore"))
            

    def search(self, query, top_k=5):
        n_docs = len(self.docs)
        if n_docs == 0:
            return[]
        avg_len = sum(self.doc_len.values()) / n_docs
        scores = Counter()


        for word in tokenize(query):
            postings = self.index.get(word, {})
            if not postings:
                continue

            # Rarity: words found in fewer documents are worth more
            idf = math.log(1 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))

            for doc_id, tf in postings.items():
                # Length: shorter documents get a small boost
                length_norm = 1 - self.b + self.b * self.doc_len[doc_id] / avg_len
                # Frequency: more occurences help, with diminishing returns
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)

        return scores.most_common(top_k)


def preview(text, length = 80):
    text = " ".join(text.split())
    return text if len(text) <= length else text[:length] + "..."

if __name__ == "__main__":
    engine = SearchEngine()
    engine.add_folder("docs")

    print(f"Indexed {len(engine.docs)} documents. Type a search, or 'quit' to exit.")

    while True:
        try:
            query = input("\nsearch> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break


        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        results = engine.search(query)
        if not results:
            print("  No results")

        for rank, (doc_id, score) in enumerate(results, start = 1):
            print(f"  {rank}. {doc_id}  ({score:.2f})")
            print(f"     {preview(engine.docs[doc_id])}")

        