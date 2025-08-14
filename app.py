from flask import Flask, jsonify
import feedparser
from transformers import pipeline
import os  # Added for port configuration

app = Flask(__name__)

# --------------------------
# 1. RSS FEEDS SETUP
# --------------------------
rss_feeds = [
    "https://www.totaltele.com/rss/news",
    "https://www.telecomtv.com/feed/",
    "https://www.telegeography.com/rss/press-releases/",
    "https://www.lightreading.com/rss_simple.asp",  
    "https://www.rcrwireless.com/rss/all",  
    "https://www.ncc.gov.ng/media-centre/press-releases/rss.xml",
    "https://nitda.gov.ng/feed/",
    "https://fmcide.gov.ng/news/feed/",
    "https://www.cbn.gov.ng/News/RssFeed.xml",
]

# --------------------------
# 2. LOAD SMALLER HUGGING FACE MODELS (Free-tier compatible)
# --------------------------
classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

# --------------------------
# 3. HELPER FUNCTIONS
# --------------------------
def fetch_rss_articles(feeds):
    articles = []
    for feed in feeds:
        parsed = feedparser.parse(feed)
        for entry in parsed.entries:
            articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", "")
            })
    return articles

def filter_telecom_laws(article):
    keywords = ["telecom", "spectrum", "regulation", "license", "operator", "telecommunications"]
    text = (article["title"] + " " + article["summary"]).lower()
    return any(k in text for k in keywords)

def classify_article(text):
    categories = ["Telecom news", "Spectrum issue", "Regulation issue", "Financial issue", "Other"]
    classification = classifier(text, candidate_labels=categories, multi_label=False)
    category = classification["labels"][0]
    return category

def is_urgent(text):
    urgency_labels = ["Urgent", "Not urgent"]
    classification = classifier(text, candidate_labels=urgency_labels, multi_label=False)
    return classification["labels"][0] == "Urgent"

def generate_summary(text, max_len=150):
    try:
        summary = summarizer(text, max_length=max_len, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except:
        return text[:max_len]

# --------------------------
# 4. ROUTE
# --------------------------
@app.route("/")
def get_urgent_issues():
    articles = fetch_rss_articles(rss_feeds)
    results = []

    for art in articles:
        if not filter_telecom_laws(art):
            continue
        
        full_text = art["title"] + ". " + art["summary"]
        if is_urgent(full_text):
            results.append({
                "issue_name": art["title"],
                "issue_summary": generate_summary(full_text),
                "issue_date": art["published"],
                "issue_source_link": art["link"]
            })

    if not results:
        return jsonify({"message": "No new issue"}), 200

    return jsonify(results), 200

# --------------------------
# 5. RUN APP (Render compatible)
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
