from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    text = text.lower()
    for ch in [",", ".", "!", "?", "(", ")", ":"]:
        text = text.replace(ch, "")
    return text


def calculate_similarity(resume, jd):
    documents = [resume, jd]

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return round(similarity * 100)


def main():
    print("\nAI Resume Matcher\n")

    resume = input("Enter Resume Text:\n")
    jd = input("\nEnter Job Description Text:\n")

    resume = clean_text(resume)
    jd = clean_text(jd)

    match_percentage = calculate_similarity(resume, jd)

    print("\n--- Result ---")
    print(f"Match Percentage: {match_percentage}%")

    if match_percentage >= 70:
        print("Status: Good Match")
    else:
        print("Status: Needs Improvement")


if __name__ == "__main__":
    main()
