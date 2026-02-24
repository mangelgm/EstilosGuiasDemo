"""
Explainability Module for Cimaprompter
Implements SHAP and LIME for local text classification model

This module demonstrates explainability techniques on a small local model
that classifies programming questions by type (loops, conditionals, variables, etc.)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import re

# For LIME
from lime.lime_text import LimeTextExplainer

# For text classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ============================================================================
# LOCAL MODEL: Programming Question Classifier
# ============================================================================

class ProgrammingQuestionClassifier:
    """
    Simple classifier that categorizes programming questions.
    This is the model we'll explain with SHAP/LIME.
    """

    def __init__(self):
        # Training data: programming questions and their categories
        self.categories = [
            'bucles',           # loops (while, for)
            'condicionales',    # if-else
            'variables',        # data types, variables
            'funciones',        # functions
            'listas',          # arrays, lists
            'general'          # general programming
        ]

        # Simple training examples
        training_texts = [
            # Bucles
            "¿Cómo funciona un bucle while?",
            "¿Qué es un ciclo for?",
            "¿Cuándo usar while en lugar de for?",
            "Explicame los bucles en Python",
            "¿Cómo hacer un loop infinito?",
            "¿Qué significa iterar?",

            # Condicionales
            "¿Qué es un if-else?",
            "¿Cómo funcionan las condiciones?",
            "¿Cuándo usar elif?",
            "Explicame los condicionales",
            "¿Qué es un statement condicional?",
            "¿Cómo evaluar una condición?",

            # Variables
            "¿Qué es una variable?",
            "¿Cuáles son los tipos de datos?",
            "¿Qué significa int, float, string?",
            "Explicame las variables en Python",
            "¿Cómo declarar una variable?",
            "¿Qué es el tipo de dato?",

            # Funciones
            "¿Qué es una función?",
            "¿Cómo definir una función?",
            "¿Qué significa return?",
            "Explicame las funciones",
            "¿Qué son los parámetros?",
            "¿Cómo llamar una función?",

            # Listas
            "¿Qué es una lista?",
            "¿Cómo usar arrays?",
            "¿Qué significa indexar?",
            "Explicame las listas en Python",
            "¿Cómo agregar elementos a una lista?",
            "¿Qué es un arreglo?",

            # General
            "¿Qué es programación?",
            "¿Cómo empezar a programar?",
            "¿Qué es Python?",
            "Explicame la lógica de programación",
            "¿Qué es un algoritmo?",
            "¿Cómo resolver este problema?"
        ]

        training_labels = (
            [0] * 6 +  # bucles
            [1] * 6 +  # condicionales
            [2] * 6 +  # variables
            [3] * 6 +  # funciones
            [4] * 6 +  # listas
            [5] * 6    # general
        )

        # Create and train pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=100, ngram_range=(1, 2))),
            ('classifier', MultinomialNB())
        ])

        self.pipeline.fit(training_texts, training_labels)

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict category for texts."""
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probability distribution for texts."""
        return self.pipeline.predict_proba(texts)

    def get_category_name(self, category_id: int) -> str:
        """Get category name from id."""
        return self.categories[category_id]

# ============================================================================
# LIME EXPLAINER
# ============================================================================

class LIMEExplainer:
    """LIME-based explainer for text classification."""

    def __init__(self, classifier: ProgrammingQuestionClassifier):
        self.classifier = classifier
        self.explainer = LimeTextExplainer(
            class_names=classifier.categories,
            random_state=42
        )

    def explain(self, text: str, num_features: int = 10) -> Dict:
        """
        Generate LIME explanation for a text.

        Returns:
            Dict with explanation details and visualization data
        """
        # Get prediction
        pred_proba = self.classifier.predict_proba([text])[0]
        pred_class = np.argmax(pred_proba)

        # Generate LIME explanation
        exp = self.explainer.explain_instance(
            text,
            self.classifier.predict_proba,
            num_features=num_features,
            top_labels=3
        )

        # Extract feature weights for the predicted class
        feature_weights = exp.as_list(label=pred_class)

        # Prepare visualization data
        words = [fw[0] for fw in feature_weights]
        weights = [fw[1] for fw in feature_weights]

        # Get highlighted text (words with their weights)
        highlighted_tokens = []
        for word, weight in feature_weights:
            # Clean the word from LIME's special characters
            clean_word = word.strip('<>=')
            highlighted_tokens.append({
                'word': clean_word,
                'weight': weight,
                'color': 'green' if weight > 0 else 'red'
            })

        return {
            'predicted_category': self.classifier.get_category_name(pred_class),
            'confidence': float(pred_proba[pred_class]),
            'probabilities': {
                self.classifier.categories[i]: float(prob)
                for i, prob in enumerate(pred_proba)
            },
            'feature_weights': list(zip(words, weights)),
            'highlighted_tokens': highlighted_tokens,
            'explanation_text': f"LIME identifica las palabras clave que influyeron en clasificar esta pregunta como '{self.classifier.get_category_name(pred_class)}'."
        }

# ============================================================================
# SHAP EXPLAINER (Simplified)
# ============================================================================

class SimplifiedSHAPExplainer:
    """
    Simplified SHAP-like explainer for text classification.

    Note: Full SHAP for text is computationally expensive.
    This provides SHAP-inspired feature importance.
    """

    def __init__(self, classifier: ProgrammingQuestionClassifier):
        self.classifier = classifier

    def explain(self, text: str) -> Dict:
        """
        Generate SHAP-inspired explanation.

        Uses permutation-based feature importance to estimate
        SHAP values (simplified approach).
        """
        # Get baseline prediction
        pred_proba = self.classifier.predict_proba([text])[0]
        pred_class = np.argmax(pred_proba)
        baseline_prob = pred_proba[pred_class]

        # Tokenize
        words = text.lower().split()

        # Calculate importance by removing each word
        word_importance = []

        for i, word in enumerate(words):
            # Create text without this word
            words_without = words[:i] + words[i+1:]
            text_without = ' '.join(words_without)

            if not text_without:
                continue

            # Get new prediction
            new_proba = self.classifier.predict_proba([text_without])[0]
            new_prob = new_proba[pred_class]

            # Importance = drop in probability
            importance = baseline_prob - new_prob

            word_importance.append((word, float(importance)))

        # Sort by absolute importance
        word_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        # Get top features
        top_features = word_importance[:10]

        return {
            'predicted_category': self.classifier.get_category_name(pred_class),
            'confidence': float(baseline_prob),
            'probabilities': {
                self.classifier.categories[i]: float(prob)
                for i, prob in enumerate(pred_proba)
            },
            'feature_importance': top_features,
            'explanation_text': f"SHAP muestra la importancia global de cada palabra en la predicción de '{self.classifier.get_category_name(pred_class)}'."
        }

# ============================================================================
# COMBINED EXPLAINER
# ============================================================================

class CombinedExplainer:
    """
    Combined LIME + SHAP explainer for programming questions.
    """

    def __init__(self):
        self.classifier = ProgrammingQuestionClassifier()
        self.lime_explainer = LIMEExplainer(self.classifier)
        self.shap_explainer = SimplifiedSHAPExplainer(self.classifier)

    def explain_question(self, question: str) -> Dict:
        """
        Generate combined explanation using both LIME and SHAP.

        Args:
            question: User's programming question

        Returns:
            Dict with both LIME and SHAP explanations
        """
        lime_result = self.lime_explainer.explain(question)
        shap_result = self.shap_explainer.explain(question)

        return {
            'question': question,
            'lime': lime_result,
            'shap': shap_result,
            'summary': {
                'category': lime_result['predicted_category'],
                'confidence': lime_result['confidence'],
                'top_lime_features': lime_result['feature_weights'][:5],
                'top_shap_features': shap_result['feature_importance'][:5]
            }
        }

    def get_visualization_data(self, explanation: Dict) -> Dict:
        """
        Prepare data for Plotly visualizations.

        Args:
            explanation: Output from explain_question()

        Returns:
            Dict with data ready for Plotly charts
        """
        lime_data = explanation['lime']
        shap_data = explanation['shap']

        # LIME bar chart data
        lime_words = [fw[0] for fw in lime_data['feature_weights'][:10]]
        lime_weights = [fw[1] for fw in lime_data['feature_weights'][:10]]

        # SHAP bar chart data
        shap_words = [fw[0] for fw in shap_data['feature_importance'][:10]]
        shap_weights = [fw[1] for fw in shap_data['feature_importance'][:10]]

        # Probability distribution data
        categories = list(lime_data['probabilities'].keys())
        probabilities = list(lime_data['probabilities'].values())

        return {
            'lime_chart': {
                'words': lime_words,
                'weights': lime_weights,
                'title': 'LIME - Importancia Local de Palabras'
            },
            'shap_chart': {
                'words': shap_words,
                'weights': shap_weights,
                'title': 'SHAP - Importancia Global de Palabras'
            },
            'probability_chart': {
                'categories': categories,
                'probabilities': probabilities,
                'title': 'Distribución de Probabilidades por Categoría'
            },
            'highlighted_text': lime_data['highlighted_tokens']
        }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_lime_chart_data(explanation: Dict) -> Tuple[List[str], List[float]]:
    """Extract LIME data for plotting."""
    lime_data = explanation['lime']
    words = [fw[0].strip('<>=') for fw in lime_data['feature_weights'][:10]]
    weights = [fw[1] for fw in lime_data['feature_weights'][:10]]
    return words, weights

def create_shap_chart_data(explanation: Dict) -> Tuple[List[str], List[float]]:
    """Extract SHAP data for plotting."""
    shap_data = explanation['shap']
    words = [fw[0] for fw in shap_data['feature_importance'][:10]]
    weights = [fw[1] for fw in shap_data['feature_importance'][:10]]
    return words, weights

def highlight_text_html(text: str, highlighted_tokens: List[Dict]) -> str:
    """
    Create HTML with highlighted words based on their importance.

    Args:
        text: Original text
        highlighted_tokens: List of dicts with 'word', 'weight', 'color'

    Returns:
        HTML string with highlighted text
    """
    result = text

    for token in highlighted_tokens:
        word = token['word']
        weight = abs(token['weight'])
        color = token['color']

        # Intensity based on weight (0-1 scale)
        intensity = min(weight * 2, 1.0)

        if color == 'green':
            bg_color = f'rgba(0, 255, 0, {intensity * 0.3})'
        else:
            bg_color = f'rgba(255, 0, 0, {intensity * 0.3})'

        # Replace word with highlighted version
        highlighted = f'<span style="background-color: {bg_color}; padding: 2px 4px; border-radius: 3px;">{word}</span>'
        result = result.replace(word, highlighted)

    return result

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the explainer
    explainer = CombinedExplainer()

    test_questions = [
        "¿Cómo funciona un bucle while en Python?",
        "¿Qué diferencia hay entre if y elif?",
        "¿Cuáles son los tipos de datos en programación?"
    ]

    print("="*60)
    print("TESTING EXPLAINABILITY MODULE")
    print("="*60)

    for question in test_questions:
        print(f"\nQuestion: {question}")
        explanation = explainer.explain_question(question)

        print(f"Category: {explanation['summary']['category']}")
        print(f"Confidence: {explanation['summary']['confidence']:.2f}")
        print(f"Top LIME features: {explanation['summary']['top_lime_features'][:3]}")
        print(f"Top SHAP features: {explanation['summary']['top_shap_features'][:3]}")
        print("-"*60)
