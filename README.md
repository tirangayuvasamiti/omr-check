# 📝 Yuva Gyan Mahotsav 2026 – OMR Auto Grader

Automatically grade OMR (Optical Mark Recognition) answer sheets for the **Yuva Gyan Mahotsav 2026** exam by Tiranga Yuva Samiti.

## 🚀 Deploy on Streamlit Cloud (GitHub)

### 1. Create a GitHub Repository

```
your-repo/
├── app.py
├── requirements.txt
├── packages.txt
└── .streamlit/
    └── config.toml
```

### 2. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit - OMR Grader"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 3. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repo and `main` branch
4. Set **Main file path** to `app.py`
5. Click **Deploy!**

---

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📋 Features

- ✅ **Automatic bubble detection** using OpenCV
- ✅ **Perspective correction** – works even with slightly angled scans
- ✅ **60-question support** (3 columns × 20 rows – matches your OMR sheet design)
- ✅ **Customizable marking scheme** (+1 / −0.25 default)
- ✅ **Multiple-mark detection** (penalizes or flags)
- ✅ **Batch processing** – grade many sheets at once
- ✅ **Export results** to CSV and JSON
- ✅ **Debug overlay** to visualize detected bubbles
- ✅ **Grade distribution chart**
- ✅ **Answer key** via manual entry, CSV, or JSON upload

---

## 📸 Scanning Tips for Best Accuracy

| Do ✅ | Avoid ❌ |
|---|---|
| Scan at 300 DPI | Low-res phone photos |
| Keep sheet flat | Wrinkled / folded sheets |
| Good even lighting | Strong shadows / glare |
| Full sheet in frame | Cropped edges |
| Dark filled bubbles | Lightly / partially filled |

---

## 📁 Answer Key Formats

**CSV:**
```
Question,Answer
1,A
2,C
3,B
```

**JSON (object):**
```json
{"1":"A","2":"C","3":"B"}
```

**JSON (array):**
```json
["A","C","B","D","A",...]
```

---

*Built for Tiranga Yuva Samiti · Yuva Gyan Mahotsav 2026*
