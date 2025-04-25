import gradio as gr
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from symspellpy import SymSpell, Verbosity
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download NLTK data
nltk.download('punkt')

# Initialize spell checker
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = 'frequency_dictionary_en_82_765.txt'  # Place this in the project directory
if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
    print("Warning: Spell check dictionary not found. Download 'frequency_dictionary_en_82_765.txt' and place it in the project directory.")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and tokenizer
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=6)
state_dict = torch.load('best_roberta_model_new_trained.pt', map_location=device, weights_only=True)
adjusted_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
model.load_state_dict(adjusted_state_dict)
model.to(device)
model.eval()

# Prediction function
def predict_score(text, model, tokenizer, device):
    encoding = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).cpu().item()
    return predicted_class + 1

# Spell checking function
def list_spelling_errors(text):
    common_errors = {
        "globbal": "global", "hoter": "hotter", "activitees": "activities", "fosil": "fossil",
        "tree's": "trees", "dyoxide": "dioxide", "tempature": "temperature", "fastly": "quickly",
        "causeing": "causing", "see levels": "sea levels", "city's": "cities", "bares": "bears",
        "disapering": "disappearing", "drougts": "droughts", "pollushion": "pollution",
        "electricty": "electricity", "hole": "whole", "there": "their", "what": "that",
        "is losing": "are losing", "people is": "people are", "places is": "places are",
        "gonna": "going to", "what":"what"
    }
    errors = []
    for error, correction in common_errors.items():
        if re.search(r'\b' + re.escape(error) + r'\b', text, re.IGNORECASE):
            errors.append((error, correction))
    words = re.findall(r'\b[\w\'-]+\b', text)
    for word in words:
        if any(word.lower() == error.lower() for error, _ in errors):
            continue
        if len(word) <= 2 or word.isdigit() or (word[0].isupper() and not word.isupper()):
            continue
        suggestions = sym_spell.lookup(word.lower(), Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions and suggestions[0].term != word.lower() and suggestions[0].distance <= 1:
            errors.append((word, suggestions[0].term))
    return errors

# Score feedback function
def get_score_feedback(score):
    feedback = {
        1: {"grade": "Poor", "description": "This essay demonstrates significant difficulty in developing and organizing ideas, with major errors in grammar, spelling, and sentence structure that severely impair communication.",
            "strengths": [], "weaknesses": ["Frequent, major errors in grammar, spelling, and mechanics", "Lack of coherent structure and organization", "Minimal development of ideas", "Limited vocabulary and word choice", "Sentence fragments and run-on sentences common"],
            "recommendations": ["Focus on mastering basic sentence structure and grammar rules", "Practice spelling common words correctly and use a dictionary", "Learn to organize ideas into simple paragraphs", "Build vocabulary through regular reading", "Practice writing complete, clear sentences"]},
        2: {"grade": "Below Average", "description": "This essay shows basic attempts at communication but contains numerous errors in grammar, spelling, and organization that interfere with clarity.",
            "strengths": ["Some attempt to address the topic"], "weaknesses": ["Frequent grammar and spelling errors", "Poor paragraph structure and organization", "Limited development of ideas", "Basic and repetitive vocabulary", "Inconsistent verb tense and subject-verb agreement"],
            "recommendations": ["Practice identifying subjects and verbs in sentences", "Work on paragraph structure with clear topic sentences", "Expand vocabulary with more precise word choices", "Review basic punctuation rules", "Focus on developing ideas with supporting details"]},
        3: {"grade": "Average", "description": "This essay addresses the topic but shows inconsistent organization and development with several errors in grammar and mechanics.",
            "strengths": ["Addresses the topic with a basic structure", "Some supporting details present"], "weaknesses": ["Inconsistent organization of ideas", "Grammar and spelling errors that occasionally obscure meaning", "Limited sentence variety", "Imprecise word choice", "Weak transitions between ideas"],
            "recommendations": ["Practice creating outlines before writing", "Vary sentence structure between simple and compound", "Add specific examples to support main points", "Proofread work for common grammar and spelling errors", "Use transitional phrases to connect ideas"]},
        4: {"grade": "Good", "description": "This essay demonstrates adequate organization and development with mostly clear language, though some errors in grammar and mechanics are present.",
            "strengths": ["Clear organization with introduction and conclusion", "Adequate development of ideas", "Generally appropriate word choice", "Some sentence variety"], "weaknesses": ["Occasional grammar and spelling errors", "Some imprecise word choices", "Limited use of complex sentence structures", "Transitions could be more effective", "Supporting details sometimes lack specificity"],
            "recommendations": ["Practice combining sentences for greater complexity", "Expand vocabulary with more precise and varied word choices", "Develop stronger transitions between paragraphs", "Add more specific supporting details and examples", "Proofread for minor grammar and spelling errors"]},
        5: {"grade": "Very Good", "description": "This essay shows strong organization, thorough development, and clear language with only minor errors in grammar and mechanics.",
            "strengths": ["Well-organized structure with effective introduction and conclusion", "Thorough development of ideas with specific examples", "Varied sentence structure", "Precise vocabulary and word choice", "Few grammar and spelling errors"], "weaknesses": ["Occasional minor errors in grammar or mechanics", "Some transitions could be strengthened", "A few instances of imprecise word choice", "Limited use of sophisticated rhetorical techniques"],
            "recommendations": ["Experiment with more complex sentence structures", "Incorporate more sophisticated vocabulary where appropriate", "Add more nuanced arguments and counterarguments", "Practice using varied rhetorical techniques", "Fine-tune grammar and proofreading skills"]},
        6: {"grade": "Excellent", "description": "This essay demonstrates sophisticated organization, insightful development, and precise language with virtually no errors in grammar and mechanics.",
            "strengths": ["Sophisticated organization with compelling introduction and conclusion", "Insightful development of ideas with specific, relevant examples", "Varied and complex sentence structures", "Precise and sophisticated vocabulary", "Strong command of grammar and mechanics", "Effective use of rhetorical techniques"], "weaknesses": ["Perhaps occasional minor lapses in word choice or style", "Very few, if any, grammatical or mechanical errors"],
            "recommendations": ["Continue developing your personal writing style", "Experiment with more advanced rhetorical techniques", "Consider exploring more complex and nuanced arguments", "Read sophisticated texts to further expand vocabulary", "Share your writing skills by helping others improve"]}
    }
    return feedback.get(score, {"grade": "Unknown", "description": "Unable to provide feedback for this score.", "strengths": [], "weaknesses": [], "recommendations": ["Please check the essay scoring system."]})

# Evaluation function for Gradio
def evaluate_essay(essay):
    if not essay.strip():
        return "Please enter an essay to evaluate.", "", "", "", ""
    
    # Predict score
    score = predict_score(essay, model, tokenizer, device)
    feedback = get_score_feedback(score)

    # Color-coded score
   
    color = "#EF4444" if score <= 2 else "#FBBF24" if score <= 4 else "#10B981"
    score_output = f"<h2 style='color: {color};'>Score: {score}/6 - {feedback['grade']}</h2>\n<p style='color: {color};'>Assessment: {feedback['description']}</p>"
    
    # Spelling errors
    spelling_output = "## 📝 Spelling Errors\n"
    spelling_errors = list_spelling_errors(essay)
    if spelling_errors:
        for i, (error, correction) in enumerate(spelling_errors, 1):
            spelling_output += f"{i}. '{error}' → '{correction}'\n"
    else:
        spelling_output += "No spelling errors found.\n"
    
    # Strengths
    strengths_output = "## 🌟 Strengths\n"
    if feedback["strengths"]:
        for i, strength in enumerate(feedback["strengths"], 1):
            strengths_output += f"{i}. {strength}\n"
    else:
        strengths_output += "No specific strengths identified.\n"
    
    # Weaknesses
    weaknesses_output = "## ⚠️ Areas for Improvement\n"
    if feedback["weaknesses"]:
        for i, weakness in enumerate(feedback["weaknesses"], 1):
            weaknesses_output += f"{i}. {weakness}\n"
    else:
        weaknesses_output += "No specific weaknesses identified.\n"
    
    # Recommendations
    recommendations_output = "## 💡 Tips for Improvement\n"
    if feedback["recommendations"]:
        for i, recommendation in enumerate(feedback["recommendations"], 1):
            recommendations_output += f"{i}. {recommendation}\n"
    else:
        recommendations_output += "No specific recommendations available.\n"
    
    return score_output, spelling_output, strengths_output, weaknesses_output, recommendations_output

# Custom CSS for modern, minimalist design with updated text color
css = """
body {
    font-family: 'Arial', sans-serif;
    background-color: #E8ECEF; /* Cool neutral tone */
    color: #2D3748; /* Dark gray for text */
}
h1, h2, h3 {
    color: #2B6CB0; /* Cool blue for headers */
}
textarea {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E0 !important;
    border-radius: 8px !important;
    padding: 10px !important;
    font-size: 16px !important;
    color: #2D3748 !important; /* Dark gray for input text */
}
button {
    background-color: #2B6CB0 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
    border: none !important;
}
button:hover {
    background-color: #1A4971 !important;
}
.output-container h2, .output-container p {
    color: inherit !important; /* Allow dynamic colors to take precedence */
}
.output-container {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-top: 20px;
    color: #2D3748 !important; /* Dark gray for output text */
    }
.output-container * {
    color: #2D3748 !important; /* Ensure all child elements (e.g., p, h2) are dark gray */
}
"""

# Gradio Interface
with gr.Blocks(title="Essay Grader", theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# 📚 Essay Grader")
    gr.Markdown("""
        ### Welcome to Essay Grader!
        Hello there! This tool uses advanced AI to evaluate your essay, giving you a score from 1 to 6, along with detailed feedback on spelling errors. Simply paste your essay below, click "Evaluate," and get instant insights to improve your writing. We’ll highlight your strengths, point out areas to work on, and offer tailored tips — all while encouraging you to refine your skills independently!
    """)
    
    # Input column at the top
    with gr.Column():
        essay_input = gr.Textbox(label="Paste Your Essay Here", lines=10, placeholder="Enter your essay text...")
        submit_btn = gr.Button("Evaluate")
    
    # Output columns in a single row underneath the input
    
    with gr.Column():
        score_output = gr.Markdown(label="Score", elem_classes="output-container")
    with gr.Column():
        spelling_output = gr.Markdown(label="Spelling Errors", elem_classes="output-container")
    with gr.Column():
        strengths_output = gr.Markdown(label="Strengths", elem_classes="output-container")
    with gr.Column():
        weaknesses_output = gr.Markdown(label="Areas for Improvement", elem_classes="output-container")
    with gr.Column():
        recommendations_output = gr.Markdown(label="Tips for Improvement", elem_classes="output-container")
    
    submit_btn.click(
        fn=evaluate_essay,
        inputs=essay_input,
        outputs=[score_output, spelling_output, strengths_output, weaknesses_output, recommendations_output]
    )

# Launch the app
demo.launch()