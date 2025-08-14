from flask import Flask, jsonify
import feedparser
from transformers import pipeline
import os
from datetime import datetime

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
# 2. LOAD MODELS WITH FALLBACKS
# --------------------------
try:
    classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    classifier = None
    summarizer = None

# --------------------------
# 3. HELPER FUNCTIONS
# --------------------------
def fetch_rss_articles(feeds):
    articles = []
    for feed in feeds:
        try:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries:
                articles.append({
                    "title": entry.get("title", "No title"),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", "#"),
                    "published": entry.get("published", str(datetime.now()))
                })
        except Exception as e:
            print(f"Failed to parse feed {feed}: {str(e)}")
    return articles

def filter_telecom_laws(article):
    keywords = ["telecom", "spectrum", "regulation", "license", "operator", "telecommunications"]
    text = (article["title"] + " " + article["summary"]).lower()
    return any(k in text for k in keywords)

def classify_article(text):
    if not classifier:
        return "Unknown (Model not loaded)"
    
    categories = ["Telecom news", "Spectrum issue", "Regulation issue", "Financial issue", "Other"]
    try:
        classification = classifier(text, candidate_labels=categories, multi_label=False)
        return classification["labels"][0]
    except:
        return "Classification failed"

def is_urgent(text):
    if not classifier:
        return False
    
    try:
        urgency_labels = ["Urgent", "Not urgent"]
        classification = classifier(text, candidate_labels=urgency_labels, multi_label=False)
        return classification["labels"][0] == "Urgent"
    except:
        return False

def generate_summary(text, max_len=150):
    if not summarizer:
        return text[:max_len] + "..." if len(text) > max_len else text
    
    try:
        summary = summarizer(text, max_length=max_len, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except:
        return text[:max_len] + "..." if len(text) > max_len else text

# --------------------------
# 4. ROUTES
# --------------------------
@app.route("/")
def get_urgent_issues():
    try:
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
            return jsonify({"message": "No urgent issues found", "status": "success"}), 200

        return jsonify({"data": results, "status": "success"}), 200

    except Exception as e:
        return jsonify({
            "message": f"Error processing request: {str(e)}",
            "status": "error"
        }), 500

@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": classifier is not None,
        "timestamp": str(datetime.now())
    })

# --------------------------
# 5. RUN APP
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
