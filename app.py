from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os  # ✅ Add this line


# Load models and encoders
clf = joblib.load('model/next_role_model.pkl')
mlb = joblib.load('model/skill_binarizer.pkl')
edu_encoder = joblib.load('model/edu_encoder.pkl')
current_role_encoder = joblib.load('model/current_role_encoder.pkl')
target_role_encoder = joblib.load('model/target_role_encoder.pkl')

# Load dataset for career path and peer benchmarking
# df = pd.read_csv('balanced_career_insight_dataset.csv')
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'balanced_career_insight_dataset.csv'))
df['skills'] = df['skills'].apply(lambda x: [s.strip() for s in x.split(',')])

# Skill Requirements Auto-generation
from collections import Counter, defaultdict

def build_skill_requirements_from_data(df, top_n=10):
    role_skills = defaultdict(list)
    for _, row in df.iterrows():
        for skill in row['skills']:
            role_skills[row['target_role']].append(skill)
    return {
        role: [skill for skill, _ in Counter(skills).most_common(top_n)]
        for role, skills in role_skills.items()
    }

role_skill_requirements = build_skill_requirements_from_data(df)

# Career Path Suggestions
transition_counts = df[['current_role', 'target_role']].value_counts().reset_index(name='count')

def get_next_roles(current_role):
    matches = transition_counts[transition_counts['current_role'] == current_role]
    if matches.empty:
        return []
    return matches[['target_role', 'count']].sort_values(by='count', ascending=False).to_dict(orient='records')

# Peer Benchmarking
from sklearn.neighbors import NearestNeighbors
skills_encoded = pd.DataFrame(mlb.transform(df['skills']), columns=mlb.classes_)
df['education_encoded'] = edu_encoder.transform(df['education'])
df['current_role_encoded'] = current_role_encoder.transform(df['current_role'])

feature_matrix = pd.concat([
    skills_encoded,
    df[['education_encoded', 'current_role_encoded', 'experience_years']]
], axis=1)

knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(feature_matrix)

def find_similar_peers(user_features):
    distances, indices = knn.kneighbors(user_features)
    return df.iloc[indices[0]][['user_id', 'current_role', 'target_role', 'skills', 'experience_years']].to_dict(orient='records')

# Skill Gap Function
def analyze_skill_gap(user_skills, target_role):
    required_skills = role_skill_requirements.get(target_role, [])
    user_skills_set = set([skill.strip() for skill in user_skills])
    required_skills_set = set(required_skills)
    return list(required_skills_set & user_skills_set), list(required_skills_set - user_skills_set)

# Format career path for display
def format_career_path(career_path_data):
    if not career_path_data:
        return "No clear career progression path identified."
    
    path_lines = []
    for i, step in enumerate(career_path_data, 1):
        path_lines.append(f"{i}. {step['target_role']} ({step['count']} transitions)")
    
    return "\n".join(path_lines)

# Format peer data for display
def format_peer_data(peer_data):
    if not peer_data:
        return "No similar peers found."
    
    peer_lines = ["user_id | current_role"]
    for peer in peer_data[:5]:  # Limit to 5 peers
        peer_lines.append(f"{peer['user_id']} | {peer['current_role']}")
    
    return "\n".join(peer_lines)

# Flask App
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def generate_insights():
    data = request.json

    try:
        # Extract input data
        user_skills = data['data'][0]  # Skills as comma-separated string
        user_edu = data['data'][1]     # Education level
        user_role = data['data'][2]    # Current role
        user_exp = data['data'][3]     # Years of experience
        
        # Parse skills from comma-separated string
        if isinstance(user_skills, str):
            user_skills = [skill.strip() for skill in user_skills.split(',')]

        # Encode features
        user_skills_vec = pd.DataFrame(mlb.transform([user_skills]), columns=mlb.classes_)
        edu_encoded = edu_encoder.transform([user_edu])[0]
        role_encoded = current_role_encoder.transform([user_role])[0]

        user_features = pd.concat([
            user_skills_vec,
            pd.DataFrame([[edu_encoded, role_encoded, user_exp]], columns=["education_encoded", "current_role_encoded", "experience_years"])
        ], axis=1)

        # 1️⃣ Predict Role
        pred_label = clf.predict(user_features)[0]
        predicted_role = target_role_encoder.inverse_transform([pred_label])[0]

        # 2️⃣ Skill Gap
        matched_skills, missing_skills = analyze_skill_gap(user_skills, predicted_role)

        # 3️⃣ Career Paths
        career_path = get_next_roles(user_role)

        # 4️⃣ Peer Benchmarking
        peer_matches = find_similar_peers(user_features)

        # Format the response as a single string
        response_text = f"🎯 Predicted Next Role: {predicted_role}\n\n"
        response_text += f"✅ Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}\n\n"
        response_text += f"❌ Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}\n\n"
        response_text += f"📈 Career Path:\n{format_career_path(career_path)}\n\n"
        response_text += f"👥 Peers:\n{format_peer_data(peer_matches)}"

        # Return in the expected format
        return jsonify({
            "data": [response_text]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "Career Insight API is live 🚀"

if __name__ == '__main__':
    app.run(debug=True)
