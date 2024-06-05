from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Error detecting language: {e}")
        return None

# Example usage #dutch: "Hallo, waar kan ik een leuke cafe vinden in het buurt van el clot? "
#sample_text = "I si tinguéssim algú que pogués recopilar i conèixer tota la informació sobre el veïnat que no està a Google, i a qui poguéssim preguntar o explicar qualsevol cosa sempre que volguéssim? "
##language = detect_language(sample_text)
#print(f"The detected language is: {language}")