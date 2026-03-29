import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import librosa
import mediapipe as mp
import numpy as np
import whisper
from deepface import DeepFace
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from moviepy.editor import VideoFileClip
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
CORS(app)


WHISPER_MODEL = None
SENTIMENT_PIPELINE = None
SEMANTIC_MODEL = None
POSE = None

# Landmark indices in MediaPipe pose outputs.
NOSE_IDX = 0
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12

FILLER_WORDS = {
    "um",
    "uh",
    "like",
    "you know",
    "actually",
    "basically",
    "literally",
    "so",
}


def _lazy_load_models() -> None:
    global WHISPER_MODEL, SENTIMENT_PIPELINE, SEMANTIC_MODEL, POSE

    if WHISPER_MODEL is None:
        # Use 'small' model for better accuracy than 'base'
        # 'base' has ~96% WER (Word Error Rate), 'small' has ~95%
        WHISPER_MODEL = whisper.load_model("small")

    if SENTIMENT_PIPELINE is None:
        SENTIMENT_PIPELINE = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )

    if SEMANTIC_MODEL is None:
        SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    if POSE is None:
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                POSE = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            else:
                # Newer mediapipe builds may expose only tasks API.
                # Keep service functional and fall back for posture scoring.
                POSE = False
                print("Warning: mediapipe.solutions.pose is unavailable; posture scoring will use fallback.")
        except Exception as exc:
            POSE = False
            print(f"Warning: failed to initialize MediaPipe pose model: {exc}")


def _clamp_score(value: float) -> int:
    return int(max(0, min(100, round(value))))


def _extract_frames(video_path: str, sample_every_n: int = 15) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    idx = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every_n == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames


def _analyze_posture(frames: List[np.ndarray]) -> int:
    if not frames:
        return 50

    if not POSE:
        # Fallback posture score when pose model is unavailable.
        return 55

    scores = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = POSE.process(rgb)
        if not result.pose_landmarks:
            continue

        landmarks = result.pose_landmarks.landmark
        ls = landmarks[LEFT_SHOULDER_IDX]
        rs = landmarks[RIGHT_SHOULDER_IDX]
        nose = landmarks[NOSE_IDX]

        shoulder_level_diff = abs(ls.y - rs.y)
        shoulder_visibility = (ls.visibility + rs.visibility) / 2.0

        # Better posture if shoulders are level, visible, and head is in-frame.
        frame_score = 100 - (shoulder_level_diff * 100)
        frame_score += shoulder_visibility * 15
        frame_score += 10 if 0.05 < nose.y < 0.9 else -10
        scores.append(_clamp_score(frame_score))

    if not scores:
        return 55

    return _clamp_score(float(np.mean(scores)))


def _analyze_facial_emotions(frames: List[np.ndarray]) -> Tuple[int, Dict[str, float]]:
    if not frames:
        return 50, {"neutral": 1.0}

    sampled = frames[: min(20, len(frames))]
    emotion_totals: Counter = Counter()
    confidence_accum = []

    for frame in sampled:
        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
            )
            if isinstance(result, list):
                result = result[0]

            emotions = result.get("emotion", {})
            dominant = result.get("dominant_emotion", "neutral")
            for k, v in emotions.items():
                emotion_totals[k] += float(v)

            confidence_accum.append(float(emotions.get(dominant, 0.0)))
        except Exception:
            continue

    if not emotion_totals:
        return 55, {"neutral": 1.0}

    total = sum(emotion_totals.values()) or 1.0
    normalized = {k: round(v / total, 3) for k, v in emotion_totals.items()}

    positive = normalized.get("happy", 0.0) + normalized.get("surprise", 0.0)
    neutral = normalized.get("neutral", 0.0)
    negative = normalized.get("angry", 0.0) + normalized.get("fear", 0.0) + normalized.get("sad", 0.0)

    facial_score = 50 + (positive * 55) + (neutral * 30) - (negative * 35)
    if confidence_accum:
        facial_score += min(10, float(np.mean(confidence_accum)) / 10)

    return _clamp_score(facial_score), normalized


def _extract_audio_to_wav(video_path: str, wav_path: str) -> bool:
    # Uses ffmpeg locally to extract mono 16kHz audio for whisper/librosa.
    cmd = (
        f'ffmpeg -y -i "{video_path}" -ac 1 -ar 16000 "{wav_path}" '
        f'>nul 2>&1'
    )
    return os.system(cmd) == 0


def _preprocess_audio(wav_path: str, output_path: str) -> bool:
    """Normalize audio for better transcription."""
    try:
        import scipy.io.wavfile as wavfile
        
        y, sr = librosa.load(wav_path, sr=16000)
        
        # Normalize audio volume
        y_normalized = librosa.util.normalize(y)
        
        # Convert to int16 and save
        wavfile.write(output_path, sr, (y_normalized * 32767).astype(np.int16))
        return True
    except Exception as e:
        print(f"Audio preprocessing skipped: {e}")
        return False


def _transcribe_audio(wav_path: str) -> str:
    """Transcribe audio using Whisper 'small' model for better accuracy."""
    try:
        # Try to preprocess for better quality
        preprocessed_path = wav_path.replace('.wav', '_processed.wav')
        audio_path = wav_path
        
        if _preprocess_audio(wav_path, preprocessed_path):
            audio_path = preprocessed_path
        
        # Transcribe with 'small' model - much better accuracy than 'base'
        result = WHISPER_MODEL.transcribe(
            audio_path,
            fp16=False,
            language="en",
            verbose=False,
        )
        
        # Clean up preprocessed file if created
        try:
            if os.path.exists(preprocessed_path) and preprocessed_path != wav_path:
                os.remove(preprocessed_path)
        except:
            pass
        
        text = result.get("text") or ""
        return text.strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""


def _count_filler_words(text: str) -> int:
    lowered = text.lower()
    count = 0
    for filler in FILLER_WORDS:
        count += len(re.findall(rf"\\b{re.escape(filler)}\\b", lowered))
    return count


def _analyze_fluency_and_audio(transcript: str, wav_path: str) -> int:
    words = re.findall(r"\b\w+\b", transcript)
    word_count = len(words)
    filler_count = _count_filler_words(transcript)

    y, sr = librosa.load(wav_path, sr=16000)
    rms = librosa.feature.rms(y=y).mean() if y.size else 0.0
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Heuristic fluency score using transcript quality + audio stability.
    base = 60
    if word_count > 80:
        base += 10
    elif word_count < 30:
        base -= 10

    if filler_count:
        base -= min(20, filler_count * 2)

    if 90 <= tempo <= 180:
        base += 8
    else:
        base -= 5

    if 0.01 <= rms <= 0.1:
        base += 7

    return _clamp_score(base)


def _analyze_sentiment(transcript: str) -> int:
    if not transcript:
        return 50

    chunks = [transcript[i : i + 400] for i in range(0, len(transcript), 400)]
    preds = SENTIMENT_PIPELINE(chunks)

    score_accum = []
    for pred in preds:
        label = pred.get("label", "NEUTRAL").upper()
        confidence = float(pred.get("score", 0.5))
        if label == "POSITIVE":
            score_accum.append(50 + confidence * 50)
        else:
            score_accum.append(50 - confidence * 30)

    return _clamp_score(float(np.mean(score_accum)))


def _recommendation(interview_score: int) -> str:
    if interview_score >= 85:
        return "Strong Hire"
    if interview_score >= 70:
        return "Consider"
    if interview_score >= 55:
        return "Needs Improvement"
    return "Not Recommended"


def _strengths_and_improvements(
    posture_score: int,
    facial_score: int,
    fluency_score: int,
    sentiment_score: int,
    transcript: str,
) -> Tuple[List[str], List[str]]:
    strengths: List[str] = []
    improvements: List[str] = []

    if posture_score >= 75:
        strengths.append("Good posture")
    else:
        improvements.append("Maintain a more upright and stable posture")

    if facial_score >= 70:
        strengths.append("Positive facial engagement")
    else:
        improvements.append("Improve facial expressiveness and eye focus")

    if fluency_score >= 75:
        strengths.append("Clear speaking fluency")
    else:
        improvements.append("Reduce filler words")

    if sentiment_score >= 70:
        strengths.append("Confident tone in responses")
    else:
        improvements.append("Use more positive and assertive phrasing")

    if len(re.findall(r"\b(project|impact|led|built|improved)\b", transcript.lower())) >= 2:
        strengths.append("Good use of impact-oriented keywords")
    else:
        improvements.append("Include more impact-focused examples")

    if not strengths:
        strengths.append("Willingness to attempt all responses")

    return strengths[:4], improvements[:4]


@app.route("/analyze-interview", methods=["POST"])
def analyze_interview():
    if "video" not in request.files:
        return jsonify({"error": "Missing multipart field: video"}), 400

    video_file = request.files["video"]
    if not video_file or not video_file.filename:
        return jsonify({"error": "No video file uploaded"}), 400

    suffix = Path(video_file.filename).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        video_path = temp_video.name
        video_file.save(video_path)

    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)

    try:
        _lazy_load_models()

        frames = _extract_frames(video_path)
        posture_score = _analyze_posture(frames)
        facial_score, emotions = _analyze_facial_emotions(frames)

        transcript = ""
        fluency_score = 55
        if _extract_audio_to_wav(video_path, wav_path):
            transcript = _transcribe_audio(wav_path)
            fluency_score = _analyze_fluency_and_audio(transcript, wav_path)

        sentiment_score = _analyze_sentiment(transcript)

        # Check if candidate actually answered
        if not transcript.strip() or len(transcript.split()) < 3:
            fluency_score = 0
            sentiment_score = 0
            interview_score = 0
            strengths = ["No speech detected"]
            areas_to_improve = ["Please provide a spoken answer to the question."]
        else:
            interview_score = _clamp_score(
                (0.30 * posture_score)
                + (0.25 * facial_score)
                + (0.25 * fluency_score)
                + (0.20 * sentiment_score)
            )

            strengths, areas_to_improve = _strengths_and_improvements(
                posture_score,
                facial_score,
                fluency_score,
                sentiment_score,
                transcript,
            )

        response = {
            "posture_score": posture_score,
            "facial_score": facial_score,
            "fluency_score": fluency_score,
            "sentiment_score": sentiment_score,
            "interview_score": interview_score,
            "recommendation": _recommendation(interview_score),
            "transcript": transcript,
            "emotions": emotions,
            "strengths": strengths,
            "areas_to_improve": areas_to_improve,
        }

        return jsonify(response)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass

        try:
            os.remove(wav_path)
        except OSError:
            pass


# Fallback questions if Ollama fails
FALLBACK_QUESTIONS = {
    "questions": [
        {
            "id": 1,
            "question": "Tell me about yourself and your experience.",
            "type": "intro",
            "time_limit": 60,
            "tip": "Keep it under 2 minutes",
        },
        {
            "id": 2,
            "question": "What technical skills from your resume are most relevant to this role?",
            "type": "technical",
            "time_limit": 90,
            "tip": "Be specific",
        },
        {
            "id": 3,
            "question": "Walk me through a technical project you are most proud of.",
            "type": "technical",
            "time_limit": 90,
            "tip": "Explain your contribution",
        },
        {
            "id": 4,
            "question": "Tell me about a time you faced a difficult challenge at work. How did you handle it?",
            "type": "behavioral",
            "time_limit": 120,
            "tip": "Use STAR format",
        },
        {
            "id": 5,
            "question": "Describe a time you worked in a team to deliver a project under pressure.",
            "type": "behavioral",
            "time_limit": 120,
            "tip": "Focus on your role",
        },
        {
            "id": 6,
            "question": "Do you have any questions for us?",
            "type": "closing",
            "time_limit": 60,
            "tip": "Prepare 2-3 questions",
        },
    ],
    "source": "fallback",
}


def call_ollama(prompt):
    """Call Ollama API with the given prompt and return parsed JSON response."""
    try:
        import requests

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.1",
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=60,
        )

        if response.status_code != 200:
            return None

        result = response.json()
        response_text = result.get("response", "")

        # Parse the JSON response
        import json

        parsed = json.loads(response_text)
        return parsed
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None


def validate_questions(data):
    """Validate that the response has correct structure."""
    if not isinstance(data, dict):
        return False
    if "questions" not in data:
        return False
    questions = data["questions"]
    if not isinstance(questions, list) or len(questions) != 6:
        return False
    for q in questions:
        required_keys = {"id", "question", "type", "time_limit", "tip"}
        if not all(key in q for key in required_keys):
            return False
    return True


@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    """Generate interview questions based on job description and resume."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        job_description = data.get("job_description")
        resume_text = data.get("resume_text")

        if not job_description or not resume_text:
            return (
                jsonify(
                    {
                        "error": "Both job_description and resume_text are required"
                    }
                ),
                400,
            )

        # Build the prompt for Ollama
        prompt = f"""
You are an expert technical interviewer. Based on the job description 
and candidate resume below, generate exactly 6 interview questions.

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume_text}

Return ONLY a valid JSON object in this exact format, nothing else:
{{
  "questions": [
    {{
      "id": 1,
      "question": "Tell me about yourself and your background.",
      "type": "intro",
      "time_limit": 60,
      "tip": "Focus on relevant experience"
    }},
    {{
      "id": 2,
      "question": "<technical question based on JD skills>",
      "type": "technical",
      "time_limit": 90,
      "tip": "Use specific examples"
    }},
    {{
      "id": 3,
      "question": "<technical question based on candidate resume>",
      "type": "technical",
      "time_limit": 90,
      "tip": "Explain your thought process"
    }},
    {{
      "id": 4,
      "question": "<behavioral question - STAR format>",
      "type": "behavioral",
      "time_limit": 120,
      "tip": "Use Situation-Task-Action-Result format"
    }},
    {{
      "id": 5,
      "question": "<behavioral question about teamwork or challenge>",
      "type": "behavioral",
      "time_limit": 120,
      "tip": "Be specific about your role"
    }},
    {{
      "id": 6,
      "question": "Do you have any questions for us about the role or company?",
      "type": "closing",
      "time_limit": 60,
      "tip": "Prepare 2-3 thoughtful questions"
    }}
  ]
}}

Rules:
- Question 1 must always be the intro/tell me about yourself
- Questions 2-3 must be technical, extracted from JD required skills
- Questions 4-5 must be behavioral using STAR format
- Question 6 must always be the closing question
- Make technical questions SPECIFIC to the job description skills
- Make behavioral questions relevant to the candidate's past experience
- Return ONLY the JSON, no explanation, no markdown, no extra text
"""

        # Call Ollama
        result = call_ollama(prompt)

        # Validate the response
        if result and validate_questions(result):
            return jsonify(result)

        # If Ollama fails or validation fails, use fallback
        print(
            "Ollama response invalid or unavailable, using fallback questions"
        )
        return jsonify(FALLBACK_QUESTIONS)

    except Exception as exc:
        print(f"Error in generate_questions: {exc}")
        # Always return fallback on error, never crash
        return jsonify(FALLBACK_QUESTIONS)


def _calculate_answer_relevance(transcript: str, question: str) -> int:
    """
    Calculate semantic similarity between answer transcript and question.
    Uses sentence-transformers BERT model for semantic matching.
    """
    try:
        if not transcript or not question:
            return 0

        # Encode both question and transcript
        question_embedding = SEMANTIC_MODEL.encode(question, convert_to_tensor=True)
        transcript_embedding = SEMANTIC_MODEL.encode(
            transcript, convert_to_tensor=True
        )

        # Calculate cosine similarity (0-1)
        similarity = util.pytorch_cos_sim(question_embedding, transcript_embedding)[
            0
        ][0].item()

        # Convert to 0-100 scale
        relevance_score = int(similarity * 100)
        return max(0, min(100, relevance_score))  # Clamp to 0-100
    except Exception as e:
        print(f"Error calculating answer relevance: {e}")
        return 50  # Default to neutral score on error


def _generate_feedback(
    posture: int,
    facial: int,
    fluency: int,
    sentiment: int,
    relevance: int,
    transcript: str,
) -> str:
    """Generate constructive feedback based on scores."""
    feedback_points = []

    if relevance >= 75:
        feedback_points.append("Your answer is highly relevant to the question")
    elif relevance >= 60:
        feedback_points.append("Your answer addresses the question reasonably well")
    else:
        feedback_points.append("Try to address the question more directly")

    if posture >= 75:
        feedback_points.append("Excellent posture and body language")
    elif posture >= 60:
        feedback_points.append("Good posture—sit up a bit more for confidence")
    else:
        feedback_points.append("Work on maintaining upright posture")

    if facial >= 75:
        feedback_points.append("Great facial engagement and eye contact")
    elif facial >= 60:
        feedback_points.append(
            "Smile occasionally to show enthusiasm and confidence"
        )
    else:
        feedback_points.append("Increase facial expressions to show engagement")

    if fluency >= 75:
        feedback_points.append("Natural speech flow with minimal filler words")
    elif fluency >= 60:
        feedback_points.append("Reduce filler words like 'um' and 'uh'")
    else:
        feedback_points.append("Slow down and reduce filler words")

    if sentiment >= 75:
        feedback_points.append("Positive, professional tone throughout")
    elif sentiment >= 60:
        feedback_points.append("Maintain a more positive and confident tone")
    else:
        feedback_points.append("Show more enthusiasm and positivity")

    # Limit to 3-4 most important feedback points
    return ". ".join(feedback_points[:3]) + "."


def _calculate_answer_relevance(transcript: str, question: str) -> int:
    """
    Calculate TF-IDF cosine similarity between question and transcript.
    """
    try:
        if not transcript or not question:
            return 50

        # Clean text
        texts = [question, transcript]
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Convert to 0-100 scale
        relevance_score = int(similarity * 100)
        return max(0, min(100, relevance_score))
    except Exception as e:
        print(f"Error calculating answer relevance: {e}")
        return 50


def _generate_response_feedback(
    posture: int,
    facial: int,
    fluency: int,
    sentiment: int,
    relevance: int,
    transcript: str,
) -> str:
    """Generate constructive feedback based on scores."""
    feedback_points = []

    if relevance >= 75:
        feedback_points.append("Your answer directly addressed the question")
    elif relevance >= 60:
        feedback_points.append("Your answer was mostly relevant to the question")
    else:
        feedback_points.append("Try to address the question more directly")

    if posture >= 75:
        feedback_points.append("Excellent posture and body language")
    elif posture >= 60:
        feedback_points.append("Sit up straighter to convey confidence")
    else:
        feedback_points.append("Work on maintaining better posture")

    if facial >= 75:
        feedback_points.append("Great facial engagement and eye contact")
    elif facial >= 60:
        feedback_points.append("Show more facial expressions and engagement")
    else:
        feedback_points.append("Increase facial expressions and eye contact")

    if fluency >= 75:
        feedback_points.append("Natural speech with minimal filler words")
    elif fluency >= 60:
        feedback_points.append("Reduce filler words like 'um', 'uh', 'like'")
    else:
        feedback_points.append("Slow down and reduce filler words significantly")

    if sentiment >= 75:
        feedback_points.append("Positive and professional tone throughout")
    elif sentiment >= 60:
        feedback_points.append("Maintain a more positive and confident tone")
    else:
        feedback_points.append("Show more enthusiasm and positivity")

    # Return top 3 feedback points
    return ". ".join(feedback_points[:3]) + "."


@app.route("/analyze-interview-response", methods=["POST"])
def analyze_interview_response():
    """Analyze a single interview response video."""
    try:
        if "video" not in request.files:
            return jsonify({"error": "Missing video file"}), 400

        video_file = request.files["video"]
        question_text = request.form.get("question", "")
        question_type = request.form.get("question_type", "general")

        if not video_file or not video_file.filename:
            return jsonify({"error": "No video file uploaded"}), 400

        # Save uploaded video temporarily
        suffix = Path(video_file.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
            video_path = temp_video.name
            video_file.save(video_path)

        # Create temp audio file path
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)

        try:
            # Step 1: Extract audio from video using moviepy
            print(f"Extracting audio from video: {video_path}")
            try:
                clip = VideoFileClip(video_path)
                if clip.audio is None:
                    print("No audio track found in video")
                    clip.close()
                    transcript = ""
                    fluency_score = 50
                else:
                    clip.audio.write_audiofile(
                        wav_path, verbose=False, logger=None
                    )
                    clip.close()

                    # Step 2: Transcribe using Whisper
                    print("Transcribing audio with Whisper...")
                    _lazy_load_models()
                    result = WHISPER_MODEL.transcribe(wav_path)
                    transcript = result.get("text", "")
                    print(f"Transcript: {transcript[:100]}...")

                    # Step 5: Analyze fluency
                    print("Analyzing speech fluency...")
                    fluency_score = _analyze_fluency_and_audio(
                        transcript, wav_path
                    )
            except Exception as e:
                print(f"Error in audio processing: {e}")
                transcript = ""
                fluency_score = 50

            # Step 3: Extract frames and analyze posture
            print("Analyzing posture...")
            _lazy_load_models()
            frames = _extract_frames(video_path)
            posture_score = _analyze_posture(frames)

            # Step 4: Analyze facial emotions
            print("Analyzing facial expressions...")
            facial_score, emotions = _analyze_facial_emotions(frames)

            # Step 6: Analyze sentiment
            print("Analyzing sentiment...")
            sentiment_score = _analyze_sentiment(transcript)

            # Step 7: Calculate answer relevance
            print("Calculating answer relevance...")
            relevance_score = _calculate_answer_relevance(
                transcript, question_text
            )

            # Check if candidate actually answered
            # We consider it unanswered if there are fewer than 3 words
            if not transcript.strip() or len(transcript.split()) < 3:
                fluency_score = 0
                sentiment_score = 0
                relevance_score = 0
                response_score = 0
                feedback = "No spoken response detected. Please ensure your microphone is working and you answer the question."
            else:
                # Step 8: Calculate weighted response score
                response_score = _clamp_score(
                    (posture_score * 0.15)
                    + (facial_score * 0.20)
                    + (fluency_score * 0.20)
                    + (sentiment_score * 0.20)
                    + (relevance_score * 0.25)
                )

                # Generate feedback
                feedback = _generate_response_feedback(
                    posture_score,
                    facial_score,
                    fluency_score,
                    sentiment_score,
                    relevance_score,
                    transcript,
                )

            response_data = {
                "question": question_text,
                "question_type": question_type,
                "transcript": transcript,
                "posture_score": posture_score,
                "facial_score": facial_score,
                "fluency_score": fluency_score,
                "sentiment_score": sentiment_score,
                "answer_relevance_score": relevance_score,
                "response_score": response_score,
                "feedback": feedback,
                "emotions": emotions,
            }

            print(f"Analysis complete. Response score: {response_score}")
            return jsonify(response_data)

        except Exception as exc:
            print(f"Error analyzing response: {exc}")
            return jsonify({"error": str(exc)}), 500

        finally:
            # Cleanup temp files
            try:
                os.remove(video_path)
                print(f"Cleaned up video file: {video_path}")
            except OSError as e:
                print(f"Error removing video file: {e}")
            try:
                os.remove(wav_path)
                print(f"Cleaned up audio file: {wav_path}")
            except OSError as e:
                print(f"Error removing audio file: {e}")

    except Exception as exc:
        print(f"Unexpected error in analyze_interview_response: {exc}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
