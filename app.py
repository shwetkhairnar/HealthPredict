from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask import abort
from datetime import datetime
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import os
from werkzeug.utils import secure_filename
from slugify import slugify
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from flask_wtf import FlaskForm

from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, ValidationError
from flask_login import LoginManager , UserMixin, login_user, logout_user, current_user , login_required
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash


# Login Form



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

#login_manager = LoginManager(app)  # This uses the imported LoginManager class
#login_manager.login_view = 'login'

#@login_manager.user_loader
#def load_user(user_id):
   #    return User.query.get(int(user_id))

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    full_name = db.Column(db.String(100))
    gender = db.Column(db.String(10))
    age = db.Column(db.Integer)
    height = db.Column(db.String(10))
    weight = db.Column(db.String(10))
    blood_group = db.Column(db.String(5))
    medical_history = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
   

class BlogPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    author = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    image_url = db.Column(db.String(200))
    slug = db.Column(db.String(300), unique=True)
    excerpt = db.Column(db.String(300))
    
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', 
                                   validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is already registered.')


# Create tables
with app.app_context():
    db.create_all()

# Sample medical dataset
def create_enhanced_dataset():
    # Expanded dataset with at least 5 examples per disease
    symptoms = [
        # Flu
        ['fever', 'cough', 'fatigue', 'headache', 'sore throat', 'muscle pain'],
        ['fever', 'cough', 'fatigue', 'body aches', 'chills'],
        ['fever', 'sore throat', 'runny nose', 'sneezing'],
        ['cough', 'headache', 'fatigue', 'nasal congestion'],
        ['fever', 'chills', 'body aches', 'fatigue'],

        # COVID-19
        ['fever', 'cough', 'difficulty breathing', 'loss of taste'],
        ['fever', 'cough', 'fatigue', 'loss of smell'],
        ['headache', 'sore throat', 'fatigue', 'diarrhea'],
        ['fever', 'cough', 'chest pain', 'shortness of breath'],
        ['fatigue', 'headache', 'loss of taste', 'nasal congestion'],

        # Allergic Rhinitis
        ['sneezing', 'runny nose', 'itchy eyes', 'nasal congestion'],
        ['itchy nose', 'sneezing', 'watery eyes'],
        ['nasal congestion', 'postnasal drip', 'itchy throat'],
        ['sneezing', 'itchy eyes', 'dark circles under eyes'],
        ['runny nose', 'itchy palate', 'nasal congestion'],

        # Asthma
        ['wheezing', 'shortness of breath', 'chest tightness', 'coughing'],
        ['coughing at night', 'difficulty breathing', 'fatigue'],
        ['shortness of breath during exercise', 'wheezing', 'chest discomfort'],
        ['persistent cough', 'trouble sleeping due to breathing issues'],
        ['rapid breathing', 'tightness in chest', 'wheezing'],

        # Diabetes Type 2
        ['increased thirst', 'frequent urination', 'fatigue', 'blurred vision'],
        ['slow-healing sores', 'frequent infections', 'hunger', 'weight loss'],
        ['numbness in hands or feet', 'dry mouth', 'itchy skin'],
        ['unexplained weight loss', 'increased hunger', 'fatigue'],
        ['blurred vision', 'frequent urination', 'increased thirst'],

        # Hypertension
        ['headache', 'shortness of breath', 'nosebleeds'],
        ['chest pain', 'dizziness', 'vision problems'],
        ['fatigue', 'irregular heartbeat', 'confusion'],
        ['pounding in chest, neck, or ears', 'nausea', 'lightheadedness'],
        ['severe anxiety', 'sweating', 'difficulty sleeping'],

        # Migraine
        ['throbbing headache', 'nausea', 'sensitivity to light'],
        ['visual disturbances', 'vomiting', 'pain on one side of head'],
        ['aura', 'dizziness', 'neck stiffness'],
        ['pulsating pain', 'blurred vision', 'sensitivity to sound'],
        ['fatigue', 'irritability', 'difficulty concentrating'],

        # Tuberculosis
        ['persistent cough', 'weight loss', 'night sweats', 'fever'],
        ['coughing up blood', 'chest pain', 'fatigue'],
        ['loss of appetite', 'chills', 'prolonged cough'],
        ['shortness of breath', 'swollen lymph nodes', 'weakness'],
        ['low-grade fever', 'productive cough', 'malaise'],

        # Anemia
        ['fatigue', 'pale skin', 'shortness of breath'],
        ['dizziness', 'cold hands and feet', 'headache'],
        ['chest pain', 'weakness', 'irregular heartbeat'],
        ['brittle nails', 'cravings for non-nutritive substances', 'poor concentration'],
        ['lightheadedness', 'pale conjunctiva', 'rapid heartbeat'],

        # Pneumonia
        ['cough with phlegm', 'fever', 'chills', 'shortness of breath'],
        ['chest pain', 'fatigue', 'nausea', 'vomiting'],
        ['rapid breathing', 'sweating', 'loss of appetite'],
        ['confusion', 'headache', 'muscle pain'],
        ['cyanosis', 'productive cough', 'pleuritic chest pain'],

        # Hepatitis B
        ['jaundice', 'dark urine', 'fatigue', 'abdominal pain'],
        ['loss of appetite', 'nausea', 'joint pain'],
        ['fever', 'vomiting', 'clay-colored stool'],
        ['muscle aches', 'itching', 'light-colored stools'],
        ['swollen liver', 'yellowing of eyes', 'weakness'],

        # Dengue Fever
        ['high fever', 'severe headache', 'pain behind eyes', 'joint pain'],
        ['muscle pain', 'skin rash', 'nausea', 'vomiting'],
        ['bleeding gums', 'easy bruising', 'fatigue'],
        ['low platelet count', 'abdominal pain', 'restlessness'],
        ['loss of appetite', 'swollen glands', 'rash'],

        # Malaria
        ['fever', 'chills', 'sweating', 'headache'],
        ['nausea', 'vomiting', 'muscle pain'],
        ['fatigue', 'abdominal pain', 'diarrhea'],
        ['anemia', 'jaundice', 'rapid breathing'],
        ['enlarged spleen', 'cough', 'body aches'],

        # Chickenpox
        ['itchy rash', 'fever', 'fatigue', 'loss of appetite'],
        ['red spots', 'blisters', 'scabs'],
        ['headache', 'sore throat', 'malaise'],
        ['abdominal pain', 'irritability', 'muscle aches'],
        ['rash starting on face and spreading', 'low-grade fever', 'tiredness'],

        # Measles
        ['high fever', 'cough', 'runny nose', 'red eyes'],
        ['koplik spots', 'rash', 'sore throat'],
        ['muscle pain', 'sensitivity to light', 'fatigue'],
        ['loss of appetite', 'swollen lymph nodes', 'diarrhea'],
        ['skin rash starting on face', 'conjunctivitis', 'dry cough'],

        # Mumps
        ['swollen salivary glands', 'fever', 'headache', 'muscle aches'],
        ['fatigue', 'loss of appetite', 'pain while chewing'],
        ['earache', 'jaw pain', 'swelling on one or both sides of face'],
        ['dry mouth', 'difficulty swallowing', 'tenderness in glands'],
        ['low-grade fever', 'nausea', 'stiff neck'],

        # Rubella
        ['rash', 'low-grade fever', 'headache', 'red eyes'],
        ['swollen lymph nodes', 'joint pain', 'runny nose'],
        ['muscle pain', 'fatigue', 'sore throat'],
        ['loss of appetite', 'mild conjunctivitis', 'itchy rash'],
        ['pink rash starting on face', 'mild fever', 'cough'],

        # Whooping Cough
        ['severe coughing fits', 'whooping sound', 'vomiting after cough'],
        ['runny nose', 'nasal congestion', 'fever'],
        ['exhaustion after coughing', 'red or blue face during cough'],
        ['sneezing', 'watery eyes', 'loss of appetite'],
        ['persistent cough', 'difficulty breathing', 'chest discomfort'],

        # Typhoid Fever
        ['high fever', 'weakness', 'stomach pain', 'headache'],
        ['loss of appetite', 'diarrhea', 'rash'],
        ['constipation', 'dry cough', 'sweating'],
        ['enlarged spleen', 'fatigue', 'chills'],
        ['abdominal tenderness', 'nosebleeds', 'confusion'],

        # Cholera
        ['watery diarrhea', 'vomiting', 'leg cramps', 'dehydration'],
        ['severe thirst', 'sunken eyes', 'dry mouth', 'low blood pressure'],
        ['rapid heart rate', 'muscle cramps', 'nausea', 'fatigue'],
        ['watery stool', 'restlessness', 'weak pulse', 'dizziness'],
        ['profuse diarrhea', 'electrolyte imbalance', 'dry skin', 'lethargy'],
    ]

    diseases = (
        ['flu'] * 5 + ['covid-19'] * 5 + ['allergic rhinitis'] * 5 +
        ['asthma'] * 5 + ['diabetes type 2'] * 5 + ['hypertension'] * 5 +
        ['migraine'] * 5 + ['tuberculosis'] * 5 + ['anemia'] * 5 +
        ['pneumonia'] * 5 + ['hepatitis b'] * 5 + ['dengue fever'] * 5 +
        ['malaria'] * 5 + ['chickenpox'] * 5 + ['measles'] * 5 +
        ['mumps'] * 5 + ['rubella'] * 5 + ['whooping cough'] * 5 +
        ['typhoid fever'] * 5 + ['cholera'] * 5
    )

    # For now, we can leave severity/urgency blank or add later
    df = pd.DataFrame({
        'symptoms': symptoms,
        'disease': diseases
    })

    return df


def get_recommendation(disease):
 recommendations = {
    'flu': 'Visit a doctor within 2 days for antiviral medications. Stay hydrated, get plenty of rest, and consider using warm compresses for muscle aches.\n**Home remedies:** Drink warm fluids like herbal teas, use a humidifier to ease nasal congestion, and inhale steam with eucalyptus oil.',

    'covid-19': 'Seek immediate medical attention, especially if symptoms worsen or breathing becomes difficult. Isolate yourself to avoid spreading the virus.\n**Home remedies:** Stay hydrated, rest, use warm salt water gargles for throat pain, and consume immunity-boosting foods like ginger, garlic, and turmeric milk.',

    'allergic rhinitis': 'Usually manageable with self-care. Use over-the-counter antihistamines, avoid allergens, and keep windows closed during high pollen times.\n**Home remedies:** Rinse nasal passages with saline solution (neti pot), consume local honey regularly, and use peppermint tea for relief.',

    'common cold': 'Self-care is usually sufficient. Get plenty of rest, drink warm fluids, and avoid cold exposure.\n**Home remedies:** Ginger tea with honey and lemon, gargle with warm salt water, and apply vapor rub to the chest.',

    'strep throat': 'See a doctor within 24 hours to get antibiotics and prevent complications. Avoid sharing utensils or drinks.\n**Home remedies:** Gargle with warm salt water, drink chamomile tea, and eat soft, soothing foods like warm soup or mashed potatoes.',

    'migraine': 'Take over-the-counter pain relievers like ibuprofen and rest in a dark, quiet room. Avoid known triggers like strong smells or bright lights.\n**Home remedies:** Apply a cold compress to the head, practice deep breathing, and drink ginger tea or peppermint tea.',

    'pneumonia': 'Seek immediate medical care, especially for elderly or immunocompromised individuals. Follow prescribed antibiotic or antiviral treatment.\n**Home remedies:** Rest, drink warm soups, use a humidifier to ease breathing, and stay hydrated.',

    'bronchitis': 'Visit a doctor within 2 days if cough persists. Use a humidifier and avoid smoking or polluted environments.\n**Home remedies:** Drink ginger-honey tea, inhale steam, and use turmeric milk at night for its anti-inflammatory benefits.',

    'asthma': 'Use prescribed inhalers during attacks and consult your doctor for long-term management. Avoid known triggers like dust and smoke.\n**Home remedies:** Breathing exercises (like pursed-lip breathing), ginger tea, and foods rich in omega-3 fatty acids can support lung health.',

    'diabetes type 2': 'Schedule regular checkups, monitor blood sugar, and maintain a balanced, low-sugar diet.\n**Home remedies:** Drink fenugreek seed water, consume cinnamon and bitter gourd juice, and walk daily to manage glucose levels.',

    'hypertension': 'Consult your physician to monitor blood pressure and adjust medication if necessary. Reduce salt intake and manage stress.\n**Home remedies:** Garlic, hibiscus tea, flaxseeds, and daily walking or meditation can help reduce blood pressure.',

    'tuberculosis': 'Seek immediate treatment through a doctor and follow the full antibiotic course.\n**Home remedies:** Eat high-protein foods, include garlic and black pepper in your diet, and get plenty of sunlight and rest.',

    'anemia': 'Take iron supplements under medical supervision. Eat iron-rich foods like leafy greens and red meat.\n**Home remedies:** Consume beetroot juice, pomegranate, dates, raisins, and jaggery regularly.',

    'hepatitis b': 'Consult a hepatologist for antiviral therapy. Avoid alcohol and maintain good hygiene.\n**Home remedies:** Eat a liver-friendly diet (low-fat, high-fiber), drink green tea, and avoid processed foods.',

    'dengue fever': 'Seek medical attention immediately. Rest and stay hydrated.\n**Home remedies:** Drink papaya leaf juice (boosts platelet count), consume coconut water, and use neem leaf tea for detox.',

    'malaria': 'Start antimalarial drugs immediately and stay in a mosquito-free environment.\n**Home remedies:** Use neem or holy basil (tulsi) tea, consume citrus fruits, and drink plenty of fluids.',

    'chickenpox': 'Usually treated at home. Isolate patient, use calamine lotion for itching, and avoid scratching.\n**Home remedies:** Oatmeal baths, baking soda in bathwater, and neem leaf paste on rashes can relieve itching.',

    'measles': 'Isolate and see a doctor immediately. Monitor symptoms and manage fever.\n**Home remedies:** Rest in a dark room, use lukewarm sponge baths, and drink plenty of orange juice or carrot juice for vitamin A.',

    'mumps': 'Rest and apply cold compresses on swollen glands. Avoid sour foods and get doctor’s consultation.\n**Home remedies:** Drink warm fluids, apply warm or cold compresses, and avoid acidic foods that cause gland pain.',

    'rubella': 'Monitor symptoms and rest. Seek medical care especially if pregnant.\n**Home remedies:** Drink herbal teas, use warm compresses for joint pain, and eat vitamin-C rich foods to boost immunity.',

    'whooping cough': 'Consult a doctor for antibiotics. Stay in a humidified environment.\n**Home remedies:** Ginger-honey mixture, turmeric milk, and steam inhalation can relieve coughing.',

    'typhoid fever': 'Seek immediate care and complete full antibiotic treatment. Avoid spicy and solid foods.\n**Home remedies:** Drink ORS, eat soft foods like khichdi or porridge, and use boiled water only.',

    'cholera': 'Urgent rehydration with ORS or IV fluids is essential. Seek antibiotics if needed.\n**Home remedies:** Drink coconut water, consume lemon juice and ginger, and maintain good hygiene practices.',

    'headache': 'Rest in a dark room, stay hydrated, and avoid eye strain.\n**Home remedies:** Apply peppermint or lavender oil on the temples, drink ginger tea, and take short naps.',

    'fatigue': 'Rest well, eat nutritious foods, and drink enough water.\n**Home remedies:** Drink ashwagandha tea, consume almonds, dates, or banana smoothies for energy.',

    'nasal congestion': 'Use a decongestant or try steam inhalation.\n**Home remedies:** Inhale steam with eucalyptus oil, drink hot soups, and use saline nasal sprays.',

    'diarrhea': 'Stay hydrated and avoid dairy or spicy food. Seek medical help if it lasts over 2 days.\n**Home remedies:** Eat boiled rice, bananas, applesauce (BRAT diet), and drink ORS solution.',

    'sore throat': 'Use throat lozenges, drink warm fluids, and rest your voice.\n**Home remedies:** Gargle with warm salt water, drink honey-lemon tea, and avoid cold beverages.',
}
 return recommendations.get(disease, 'consult a healthcare provider')

# Initialize sample data
def init_sample_data():
    # Create admin user if not exists
    if not User.query.filter_by(is_admin=True).first():
        admin = User(
            username='admin',
            email='admin@healthpredict.com',
            password=bcrypt.generate_password_hash('admin123').decode('utf-8'),
            full_name='Admin User',
            gender='other',
            age=30,
            height='170 cm',
            weight='70 kg',
            blood_group='O+',
            is_admin=True
        )
        db.session.add(admin)
        db.session.commit()

    # Create sample blogs if none exist
    if not BlogPost.query.first():
        sample_blogs = [
           BlogPost(
    title="Understanding Blood Pressure: The Complete Guide",
    content="""
    <div class="blog-content">
        <h2 class="text-primary">What is Blood Pressure?</h2>
        <p>Blood pressure is the force exerted by circulating blood against the walls of your arteries, the vessels that carry blood from your heart to the rest of your body. It's one of the most important vital signs doctors use to assess your overall health.</p>
        
        <div class="row my-4">
            <div class="col-md-6">
                <div class="card border-primary">
                    <div class="card-header bg-primary text-white">
                        <h5><i class="fas fa-heartbeat"></i> Systolic Pressure</h5>
                    </div>
                    <div class="card-body">
                        <p>The top number (e.g., 120 in "120/80") measures the pressure in your arteries when your heart beats.</p>
                        <ul>
                            <li>Normal: Below 120 mmHg</li>
                            <li>Elevated: 120-129 mmHg</li>
                            <li>High: 130+ mmHg</li>
                        </ul>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card border-info">
                    <div class="card-header bg-info text-white">
                        <h5><i class="fas fa-heart"></i> Diastolic Pressure</h5>
                    </div>
                    <div class="card-body">
                        <p>The bottom number measures the pressure between heartbeats when your heart is resting.</p>
                        <ul>
                            <li>Normal: Below 80 mmHg</li>
                            <li>Elevated: 80-89 mmHg</li>
                            <li>High: 90+ mmHg</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <h3 class="mt-5">Why Blood Pressure Matters</h3>
        <p>Chronic high blood pressure (hypertension) is often called the "silent killer" because it typically has no symptoms but can lead to:</p>
        
        <div class="alert alert-danger">
            <h4><i class="fas fa-exclamation-triangle"></i> Health Risks</h4>
            <div class="row">
                <div class="col-md-6">
                    <ul>
                        <li>Heart attack and heart disease</li>
                        <li>Stroke</li>
                        <li>Aneurysms</li>
                    </ul>
                </div>
                <div class="col-md-6">
                    <ul>
                        <li>Kidney damage</li>
                        <li>Vision loss</li>
                        <li>Memory problems</li>
                    </ul>
                </div>
            </div>
        </div>

        <h3 class="mt-5">Comprehensive Management Strategies</h3>
        
        <div class="card mb-4">
            <div class="card-header bg-success text-white">
                <h4><i class="fas fa-utensils"></i> Dietary Approaches</h4>
            </div>
            <div class="card-body">
                <h5>The DASH Diet</h5>
                <p>The Dietary Approaches to Stop Hypertension (DASH) eating plan:</p>
                <ul>
                    <li>Rich in fruits, vegetables, and whole grains</li>
                    <li>Includes fat-free or low-fat dairy products</li>
                    <li>Limits foods high in saturated fat and sugar</li>
                </ul>
                
                <h5 class="mt-3">Key Nutrients</h5>
                <div class="row">
                    <div class="col-md-4">
                        <strong>Potassium</strong>
                        <p>Helps balance sodium levels (bananas, spinach, sweet potatoes)</p>
                    </div>
                    <div class="col-md-4">
                        <strong>Magnesium</strong>
                        <p>Helps blood vessels relax (almonds, spinach, black beans)</p>
                    </div>
                    <div class="col-md-4">
                        <strong>Calcium</strong>
                        <p>Important for vascular contraction (dairy, leafy greens)</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="card mb-4">
            <div class="card-header bg-warning text-dark">
                <h4><i class="fas fa-running"></i> Lifestyle Modifications</h4>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h5>Exercise Recommendations</h5>
                        <ul>
                            <li>150 minutes/week moderate activity</li>
                            <li>Strength training 2x/week</li>
                            <li>Yoga for stress reduction</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h5>Other Lifestyle Factors</h5>
                        <ul>
                            <li>Limit alcohol to 1 drink/day (women) or 2 drinks/day (men)</li>
                            <li>Quit smoking</li>
                            <li>7-9 hours of quality sleep nightly</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <h3 class="mt-5">When to Seek Medical Help</h3>
        <div class="alert alert-info">
            <p>Consult your doctor if:</p>
            <ul>
                <li>Your blood pressure is consistently above 130/80 mmHg</li>
                <li>You experience headaches, dizziness, or nosebleeds</li>
                <li>You have risk factors like diabetes or family history</li>
            </ul>
        </div>
    </div>
    """,
    excerpt="A comprehensive guide to understanding, monitoring, and managing your blood pressure for optimal health.",
    category="Cardiology",
    slug="understanding-blood-pressure",
    author="Dr. Sarah Johnson"

            ),
          BlogPost(
    title="Nutrition Myths Debunked: Science-Based Truths",
    content="""
    <div class="blog-content">
        <h2 class="text-primary">The Truth About Popular Nutrition Myths</h2>
        <p>In an era of information overload, nutrition myths spread rapidly. This article separates fact from fiction with evidence-based information.</p>
        
        <div class="myth-section mb-5">
            <div class="card myth-card mb-4">
                <div class="card-header bg-danger text-white">
                    <h3><i class="fas fa-times-circle"></i> Myth: Carbs Are the Enemy</h3>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-8">
                            <h4 class="text-success">The Truth</h4>
                            <p>Not all carbohydrates are created equal. While refined carbs (white bread, pastries) should be limited, complex carbohydrates are essential for health:</p>
                            <ul>
                                <li>Whole grains provide sustained energy</li>
                                <li>Fiber supports digestive health</li>
                                <li>Phytonutrients in plant foods have antioxidant properties</li>
                            </ul>
                        </div>
                        <div class="col-md-4">
                            <div class="card bg-light">
                                <div class="card-body text-center">
                                    <h5>Healthy Carb Sources</h5>
                                    <ul class="list-unstyled">
                                        <li>Quinoa</li>
                                        <li>Sweet potatoes</li>
                                        <li>Legumes</li>
                                        <li>Fruits</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card myth-card mb-4">
                <div class="card-header bg-danger text-white">
                    <h3><i class="fas fa-times-circle"></i> Myth: Detox Diets Cleanse Your Body</h3>
                </div>
                <div class="card-body">
                    <h4 class="text-success">The Truth</h4>
                    <p>Your liver, kidneys, and digestive system naturally detoxify your body. Instead of extreme detoxes:</p>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <h5>Support Natural Detoxification</h5>
                            <ul>
                                <li>Stay hydrated with water</li>
                                <li>Eat fiber-rich foods</li>
                                <li>Consume cruciferous vegetables</li>
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <h5>Avoid Detox Pitfalls</h5>
                            <ul>
                                <li>Juice cleanses lack protein and fiber</li>
                                <li>Detox teas may contain laxatives</li>
                                <li>Extreme fasting can be dangerous</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <h3 class="mt-5">Evidence-Based Nutrition Principles</h3>
        
        <div class="card mb-4">
            <div class="card-header bg-success text-white">
                <h4><i class="fas fa-check-circle"></i> Fundamentals of Healthy Eating</h4>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h5>Macronutrient Balance</h5>
                        <div class="nutrition-facts">
                            <p><strong>Protein:</strong> 10-35% of calories (lean meats, fish, legumes)</p>
                            <p><strong>Carbs:</strong> 45-65% of calories (focus on whole foods)</p>
                            <p><strong>Fats:</strong> 20-35% of calories (healthy unsaturated fats)</p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h5>Micronutrient Density</h5>
                        <p>Prioritize foods rich in:</p>
                        <ul>
                            <li>Vitamins (A, C, D, E, K, B-complex)</li>
                            <li>Minerals (iron, calcium, magnesium, zinc)</li>
                            <li>Antioxidants (berries, leafy greens)</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <h3 class="mt-5">Practical Nutrition Tips</h3>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card h-100">
                    <div class="card-body">
                        <h5><i class="fas fa-shopping-basket"></i> Smart Grocery Shopping</h5>
                        <ul>
                            <li>Shop the perimeter for fresh foods</li>
                            <li>Read nutrition labels carefully</li>
                            <li>Choose whole foods over processed</li>
                        </ul>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card h-100">
                    <div class="card-body">
                        <h5><i class="fas fa-utensils"></i> Healthy Cooking Methods</h5>
                        <ul>
                            <li>Steaming preserves nutrients</li>
                            <li>Grilling adds flavor without excess fat</li>
                            <li>Stir-frying with healthy oils</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <div class="alert alert-info mt-4">
            <h4><i class="fas fa-lightbulb"></i> Key Takeaway</h4>
            <p>Instead of following fad diets, focus on balanced, varied eating patterns that you can maintain long-term. Small, sustainable changes lead to lasting health benefits.</p>
        </div>
    </div>
    """,
    excerpt="Separating nutrition facts from fiction with comprehensive, science-backed information.",
    category="Nutrition",
    slug="nutrition-myths-debunked",
    author="Nutritionist Mark Lee"
),

           BlogPost(
    title="The Science of Sleep: Why Rest Matters",
    content="""
    <div class="blog-content">
        <h2 class="text-primary">Understanding Sleep Fundamentals</h2>
        <p>Sleep is not merely "downtime" but an active state essential for physical restoration, mental health, and cognitive function.</p>
        
        <div class="row my-4">
            <div class="col-md-6">
                <div class="card border-primary">
                    <div class="card-header bg-primary text-white">
                        <h5><i class="fas fa-brain"></i> Sleep Architecture</h5>
                    </div>
                    <div class="card-body">
                        <p>A complete sleep cycle lasts about 90 minutes and includes:</p>
                        <ul>
                            <li><strong>NREM Stage 1:</strong> Light sleep (5-10 mins)</li>
                            <li><strong>NREM Stage 2:</strong> Body temperature drops (20 mins)</li>
                            <li><strong>NREM Stage 3:</strong> Deep sleep (30 mins)</li>
                            <li><strong>REM Sleep:</strong> Brain activity increases (10-60 mins)</li>
                        </ul>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card border-info">
                    <div class="card-header bg-info text-white">
                        <h5><i class="fas fa-clock"></i> Recommended Sleep Duration</h5>
                    </div>
                    <div class="card-body">
                        <ul>
                            <li>Newborns (0-3 months): 14-17 hours</li>
                            <li>Adults (18-64 years): 7-9 hours</li>
                            <li>Older adults (65+ years): 7-8 hours</li>
                        </ul>
                        <p class="mt-2">Quality matters as much as quantity - uninterrupted sleep is most restorative.</p>
                    </div>
                </div>
            </div>
        </div>

        <h3 class="mt-5">The Health Impacts of Sleep</h3>
        
        <div class="card mb-4">
            <div class="card-header bg-success text-white">
                <h4><i class="fas fa-heartbeat"></i> Physical Health Benefits</h4>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-4">
                        <h5>Immune Function</h5>
                        <p>During sleep, your body produces cytokines that help fight infection.</p>
                    </div>
                    <div class="col-md-4">
                        <h5>Metabolic Health</h5>
                        <p>Sleep regulates hormones that control appetite (ghrelin and leptin).</p>
                    </div>
                    <div class="col-md-4">
                        <h5>Cardiovascular Health</h5>
                        <p>Chronic sleep deprivation increases risk of heart disease by 48%.</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="card mb-4">
            <div class="card-header bg-warning text-dark">
                <h4><i class="fas fa-brain"></i> Cognitive Benefits</h4>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h5>Memory Consolidation</h5>
                        <p>During REM sleep, your brain processes and stores new information.</p>
                    </div>
                    <div class="col-md-6">
                        <h5>Creativity & Problem-Solving</h5>
                        <p>Sleep enhances cognitive flexibility and innovative thinking.</p>
                    </div>
                </div>
            </div>
        </div>

        <h3 class="mt-5">Comprehensive Sleep Improvement Strategies</h3>
        
        <div class="card mb-4">
            <div class="card-header bg-primary text-white">
                <h4><i class="fas fa-bed"></i> Sleep Hygiene Essentials</h4>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h5>Environment Optimization</h5>
                        <ul>
                            <li>Keep room temperature between 60-67°F (15-19°C)</li>
                            <li>Use blackout curtains to eliminate light</li>
                            <li>Consider white noise machines if needed</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h5>Pre-Sleep Routine</h5>
                        <ul>
                            <li>Establish a consistent bedtime</li>
                            <li>Limit screen time 1 hour before bed</li>
                            <li>Practice relaxation techniques</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <div class="alert alert-info">
            <h4><i class="fas fa-lightbulb"></i> When to Seek Help</h4>
            <p>Consult a sleep specialist if you experience:</p>
            <ul>
                <li>Chronic insomnia (difficulty falling/staying asleep)</li>
                <li>Excessive daytime sleepiness</li>
                <li>Symptoms of sleep apnea (loud snoring, gasping for air)</li>
            </ul>
        </div>
    </div>
    """,
    excerpt="A deep dive into the science of sleep and practical strategies for better rest.",
    category="Wellness",
    slug="science-of-sleep",
    author="Dr. Emily Chen"
)
        ]
        db.session.add_all(sample_blogs)
        db.session.commit()
        

# Train and save disease prediction model
# Advanced symptom processing
# Replace the SymptomProcessor class with this version
class SymptomProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=self.tokenize,
            preprocessor=self.preprocess,
            token_pattern=None
        )
    
    @staticmethod
    def tokenize(symptoms_str):
        return symptoms_str.split()
    
    @staticmethod
    def preprocess(symptoms_str):
        return symptoms_str
    
    def fit(self, X, y=None):
        symptoms_strs = [' '.join(symptoms) for symptoms in X]
        self.vectorizer.fit(symptoms_strs)
        return self
        
    def transform(self, X):
        symptoms_strs = [' '.join(symptoms) for symptoms in X]
        return self.vectorizer.transform(symptoms_strs)

# Enhanced model training
def train_advanced_model():
    df = create_enhanced_dataset()
    
    # Create pipeline with probability calibration
    pipeline = Pipeline([
        ('processor', SymptomProcessor()),
        ('clf', CalibratedClassifierCV(
            LinearSVC(dual=False),
            method='sigmoid',
            cv=3
        ))
    ])
    
    X = df['symptoms']
    y = df['disease']
    
    pipeline.fit(X, y)
    
    with open('advanced_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    return pipeline

# Initialize data
with app.app_context():
    init_sample_data()
    train_advanced_model()

# Helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Context processors
@app.context_processor
def inject_common_data():
    categories = db.session.query(BlogPost.category).distinct().all()
    recent_blogs = BlogPost.query.order_by(BlogPost.created_at.desc()).limit(3).all()
    return dict(
        categories=[c[0] for c in categories],
        recent_blogs=recent_blogs,
        current_year=datetime.now().year
    )

# Routes
# app.py

# ... (previous imports and setup)

# Main Home Route (keep only one)
@app.route('/')
def home():
    featured_blogs = BlogPost.query.order_by(BlogPost.created_at.desc()).limit(2).all()
    return render_template('index.html', 
                         featured_blogs=featured_blogs,
                         featured_articles=FEATURED_ARTICLES[:3])

# Featured Articles Routes (new)
@app.route('/featured-articles')
def list_featured_articles():  # Changed function name
    return render_template('featured_articles.html', articles=FEATURED_ARTICLES)

@app.route('/featured-article/<string:slug>')
def show_featured_article(slug):  # Changed function name
    article = next((a for a in FEATURED_ARTICLES if a['slug'] == slug), None)
    if not article:
        abort(404)
    return render_template('featured_article.html', article=article)

# ... (rest of your routes)

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('home'))  # Changed from 'dashboard' to 'home'
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['email'] = user.email
            session['is_admin'] = user.is_admin
            
            flash('Login successful!', 'success')
            return redirect(url_for('home'))  # Changed from 'dashboard' to 'home'
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Simple validation
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
        elif User.query.filter((User.email == email) | (User.username == username)).first():
            flash('Email or username already exists', 'danger')
        else:
            # Create new user
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            new_user = User(
                username=username,
                email=email,
                password=hashed_password
            )
            
            db.session.add(new_user)
            db.session.commit()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    # Clear all session variables
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))



# Profile routes
# Profile routes
@app.route('/profile')
def profile():
    if 'user_id' not in session:
        flash('Please login to view your profile', 'warning')
        return redirect(url_for('login', next=request.url))
    
    user = User.query.get(session['user_id'])
    if not user:
        flash('User not found', 'danger')
        return redirect(url_for('logout'))
    
    return render_template('profile.html', user=user)

@app.route('/update_medical_history', methods=['POST'])
def update_medical_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        return redirect(url_for('logout'))
    
    user.medical_history = request.form.get('medical_history', '')
    db.session.commit()
    
    flash('Medical history updated successfully!', 'success')
    return redirect(url_for('profile'))

# Blog routes
@app.route('/blogs')
def blogs():
    page = request.args.get('page', 1, type=int)
    blogs = BlogPost.query.order_by(BlogPost.created_at.desc()).paginate(page=page, per_page=5)
    return render_template('blogs.html', blogs=blogs)

@app.route('/blog/<string:slug>')
def blog_detail(slug):
    blog = BlogPost.query.filter_by(slug=slug).first_or_404()
    related_blogs = BlogPost.query.filter(
        BlogPost.category == blog.category,
        BlogPost.id != blog.id
    ).order_by(db.func.random()).limit(3).all()
    return render_template('blog_detail.html', blog=blog, related_blogs=related_blogs)

@app.route('/blogs/category/<string:category>')
def blogs_by_category(category):
    page = request.args.get('page', 1, type=int)
    blogs = BlogPost.query.filter_by(category=category)\
               .order_by(BlogPost.created_at.desc())\
               .paginate(page=page, per_page=5)
    return render_template('blogs.html', blogs=blogs, category=category)

# Admin routes
@app.route('/admin/create-blog', methods=['GET', 'POST'])
def create_blog():
    if 'user_id' not in session or not session.get('is_admin'):
        flash('Admin access required', 'danger')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        author = request.form['author']
        category = request.form['category']
        excerpt = request.form['excerpt']
        
        # Handle file upload
        image_url = None
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                image_url = f"static/uploads/{filename}"
        
        blog = BlogPost(
            title=title,
            content=content,
            author=author,
            category=category,
            excerpt=excerpt,
            slug=slugify(title),
            image_url=image_url
        )
        
        db.session.add(blog)
        db.session.commit()
        
        flash('Blog post created successfully!', 'success')
        return redirect(url_for('blog_detail', slug=blog.slug))
    
    return render_template('admin/create_blog.html')

# Add these new routes to your existing app.py

# Featured Articles Data (or you can store in database)
FEATURED_ARTICLES = [
    {
        'id': 1,
        'slug': 'balanced-diet-essentials',
        'title': 'The Complete Guide to Balanced Nutrition',
        'content': """
        <div class="article-content">
            <h2 class="text-primary">Foundations of a Healthy Diet</h2>
            <p>Proper nutrition is the cornerstone of good health. A balanced diet provides essential nutrients that your body needs to function optimally:</p>
            
            <div class="row my-4">
                <div class="col-md-6">
                    <div class="card border-success">
                        <div class="card-header bg-success text-white">
                            <h5><i class="fas fa-apple-alt"></i> Macronutrients</h5>
                        </div>
                        <div class="card-body">
                            <ul>
                                <li><strong>Proteins:</strong> 10-35% of calories (lean meats, legumes, dairy)</li>
                                <li><strong>Carbohydrates:</strong> 45-65% of calories (whole grains, fruits, vegetables)</li>
                                <li><strong>Fats:</strong> 20-35% of calories (avocados, nuts, olive oil)</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card border-info">
                        <div class="card-header bg-info text-white">
                            <h5><i class="fas fa-vitamins"></i> Micronutrients</h5>
                        </div>
                        <div class="card-body">
                            <ul>
                                <li><strong>Vitamins:</strong> A, B-complex, C, D, E, K</li>
                                <li><strong>Minerals:</strong> Calcium, iron, magnesium, zinc</li>
                                <li><strong>Antioxidants:</strong> Flavonoids, carotenoids</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <h3 class="mt-5">Building Your Plate</h3>
            <div class="plate-visual text-center mb-4">
                <div class="d-flex justify-content-center">
                    <div class="plate-circle">
                        <div class="plate-section veg" style="transform: rotate(0deg) skewY(60deg);">
                            <span>Vegetables 40%</span>
                        </div>
                        <div class="plate-section grains" style="transform: rotate(72deg) skewY(60deg);">
                            <span>Whole Grains 30%</span>
                        </div>
                        <div class="plate-section protein" style="transform: rotate(144deg) skewY(60deg);">
                            <span>Protein 20%</span>
                        </div>
                        <div class="plate-section fruit" style="transform: rotate(216deg) skewY(60deg);">
                            <span>Fruits 10%</span>
                        </div>
                    </div>
                </div>
            </div>

            <h3>Special Dietary Considerations</h3>
            <div class="row">
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <h5><i class="fas fa-heart"></i> Heart Health</h5>
                            <ul>
                                <li>Limit saturated fats to <7% of calories</li>
                                <li>Increase omega-3 fatty acids</li>
                                <li>25-30g fiber daily</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <h5><i class="fas fa-bolt"></i> Energy Needs</h5>
                            <ul>
                                <li>Athletes: 2,500-3,500 kcal/day</li>
                                <li>Sedentary adults: 1,600-2,400 kcal/day</li>
                                <li>40% carbs, 30% protein, 30% fat for active individuals</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <h5><i class="fas fa-allergies"></i> Food Sensitivities</h5>
                            <ul>
                                <li>Gluten-free alternatives</li>
                                <li>Lactose intolerance solutions</li>
                                <li>Low-FODMAP options</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <div class="alert alert-info mt-4">
                <h4><i class="fas fa-lightbulb"></i> Practical Tips</h4>
                <ul>
                    <li>Meal prep on weekends saves time</li>
                    <li>Use smaller plates for portion control</li>
                    <li>Stay hydrated with 2-3L water daily</li>
                </ul>
            </div>
        </div>
        """,
        'category': 'Nutrition',
        'image': 'nutrition.jpg',
        'author': 'Dr. Sarah Johnson',
        'date': '2023-05-15'
    },
    {
        'id': 2,
        'slug': 'home-workout-guide',
        'title': 'The Ultimate Home Fitness Program',
        'content': """
        <div class="article-content">
            <h2 class="text-primary">Comprehensive Home Exercise System</h2>
            <p>You don't need a gym membership to achieve peak fitness. This complete guide covers everything from beginner routines to advanced techniques.</p>
            
            <div class="row my-4">
                <div class="col-md-4">
                    <div class="card border-danger">
                        <div class="card-header bg-danger text-white">
                            <h5><i class="fas fa-fire"></i> Beginner (Weeks 1-4)</h5>
                        </div>
                        <div class="card-body">
                            <ul>
                                <li>Bodyweight squats: 3x10</li>
                                <li>Wall push-ups: 3x8</li>
                                <li>Plank: 30 seconds</li>
                                <li>Walking: 20 mins/day</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card border-warning">
                        <div class="card-header bg-warning text-dark">
                            <h5><i class="fas fa-bolt"></i> Intermediate (Weeks 5-8)</h5>
                        </div>
                        <div class="card-body">
                            <ul>
                                <li>Pistol squats: 3x8 per leg</li>
                                <li>Standard push-ups: 3x12</li>
                                <li>Burpees: 3x10</li>
                                <li>Jump rope: 10 mins</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card border-success">
                        <div class="card-header bg-success text-white">
                            <h5><i class="fas fa-trophy"></i> Advanced (Weeks 9+)</h5>
                        </div>
                        <div class="card-body">
                            <ul>
                                <li>One-arm push-ups: 3x5</li>
                                <li>Plyometric jumps: 3x15</li>
                                <li>Handstand push-ups: 3x5</li>
                                <li>HIIT circuits</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <h3 class="mt-5">Exercise Demonstrations</h3>
            <div class="row">
                <div class="col-md-6">
                    <div class="embed-responsive embed-responsive-16by9 mb-3">
                        <iframe class="embed-responsive-item" src="https://www.youtube.com/embed/example1" allowfullscreen></iframe>
                    </div>
                    <h5>Perfect Push-Up Form</h5>
                </div>
                <div class="col-md-6">
                    <div class="embed-responsive embed-responsive-16by9 mb-3">
                        <iframe class="embed-responsive-item" src="https://www.youtube.com/embed/example2" allowfullscreen></iframe>
                    </div>
                    <h5>Squat Masterclass</h5>
                </div>
            </div>

            <h3 class="mt-5">Equipment-Free Progression</h3>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Exercise</th>
                        <th>Beginner</th>
                        <th>Intermediate</th>
                        <th>Advanced</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Push-ups</td>
                        <td>Wall push-ups</td>
                        <td>Knee push-ups</td>
                        <td>One-arm push-ups</td>
                    </tr>
                    <tr>
                        <td>Squats</td>
                        <td>Chair-assisted</td>
                        <td>Bodyweight</td>
                        <td>Pistol squats</td>
                    </tr>
                    <tr>
                        <td>Core</td>
                        <td>Plank (knees)</td>
                        <td>Standard plank</td>
                        <td>Dragon flags</td>
                    </tr>
                </tbody>
            </table>

            <div class="alert alert-success mt-4">
                <h4><i class="fas fa-heartbeat"></i> Health Benefits</h4>
                <div class="row">
                    <div class="col-md-6">
                        <ul>
                            <li>30% lower heart disease risk</li>
                            <li>Improved bone density</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <ul>
                            <li>Reduced stress and anxiety</li>
                            <li>Better sleep quality</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """,
        'category': 'Fitness',
        'image': 'exercise.jpg',
        'author': 'Trainer Mark Wilson',
        'date': '2023-06-20'
    },
    {
        'id': 3,
        'slug': 'stress-management',
        'title': 'Modern Stress Reduction Techniques',
        'content': """
        <div class="article-content">
            <h2 class="text-primary">The Science of Stress Relief</h2>
            <p>Chronic stress affects 77% of adults. Learn evidence-based techniques to manage stress effectively.</p>
            
            <div class="card mb-4">
                <div class="card-header bg-primary text-white">
                    <h4><i class="fas fa-brain"></i> Neurological Impact</h4>
                </div>
                <div class="card-body">
                    <p>Stress triggers cortisol release which in excess can:</p>
                    <ul>
                        <li>Shrink prefrontal cortex (decision-making)</li>
                        <li>Enlarge amygdala (fear response)</li>
                        <li>Disrupt hippocampal neurogenesis (memory)</li>
                    </ul>
                </div>
            </div>

            <h3>Evidence-Based Techniques</h3>
            <div class="row">
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <h5><i class="fas fa-spa"></i> Mindfulness</h5>
                            <ul>
                                <li>Daily 10-minute meditation</li>
                                <li>Body scan exercises</li>
                                <li>Reduces anxiety by 39%</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <h5><i class="fas fa-lungs"></i> Breathing Methods</h5>
                            <ul>
                                <li>4-7-8 technique</li>
                                <li>Box breathing</li>
                                <li>Lowers blood pressure</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <h5><i class="fas fa-running"></i> Physical Activity</h5>
                            <ul>
                                <li>Yoga reduces cortisol by 26%</li>
                                <li>30-min walk decreases tension</li>
                                <li>Dancing boosts endorphins</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <h3 class="mt-5">Daily Stress Reduction Plan</h3>
            <table class="table table-bordered">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Activity</th>
                        <th>Duration</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Morning</td>
                        <td>Gratitude journaling</td>
                        <td>5 minutes</td>
                    </tr>
                    <tr>
                        <td>Midday</td>
                        <td>Walking meeting</td>
                        <td>15 minutes</td>
                    </tr>
                    <tr>
                        <td>Evening</td>
                        <td>Progressive muscle relaxation</td>
                        <td>10 minutes</td>
                    </tr>
                </tbody>
            </table>

            <div class="alert alert-info mt-4">
                <h4><i class="fas fa-mobile-alt"></i> Helpful Apps</h4>
                <div class="row">
                    <div class="col-md-4">
                        <strong>Headspace</strong>
                        <p>Guided meditations</p>
                    </div>
                    <div class="col-md-4">
                        <strong>MyFitnessPal</strong>
                        <p>Nutrition tracking</p>
                    </div>
                    <div class="col-md-4">
                        <strong>Forest</strong>
                        <p>Focus timer</p>
                    </div>
                </div>
            </div>
        </div>
        """,
        'category': 'Mental Health',
        'image': 'stress-relief.jpg',
        'author': 'Dr. Emily Chen',
        'date': '2023-07-15'
    }
]

@app.route('/about')
def about():
    return render_template('about.html')  # Make sure this template exists

@app.route('/health-tips')
def health_tips():
    return render_template('health_tips.html')  # Make sure this template exists

# Featured Articles Routes
@app.route('/featured-articles')
def featured_articles():
    return render_template('featured_articles.html', articles=FEATURED_ARTICLES)

@app.route('/featured-article/<string:slug>')
def featured_article(slug):
    article = next((a for a in FEATURED_ARTICLES if a['slug'] == slug), None)
    if not article:
        abort(404)
    return render_template('featured_article.html', article=article)

# Update home route to include featured articles

# Medical prediction routes
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        
        with open('advanced_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        probas = model.predict_proba([symptoms])[0]
        diseases = model.classes_
        
        top3_idx = np.argsort(probas)[-3:][::-1]
        predictions = [
            {
                'disease': diseases[i],
                'probability': float(probas[i]),
                # 'severity': get_severity(diseases[i]),
                'recommendation': get_recommendation(diseases[i])  # Use the function instead
            } for i in top3_idx
        ]
        
        return render_template('advanced_results.html', 
                            symptoms=symptoms,
                            predictions=predictions)
    
    df = create_enhanced_dataset()
    all_symptoms = sorted(list(set(symptom for sublist in df['symptoms'] for symptom in sublist)))
    return render_template('advanced_predict.html', symptoms=all_symptoms)

# Helper functions
#def get_severity(disease):
  #  df = create_enhanced_dataset()
   # return df[df['disease'] == disease]['severity'].values[0]

# def get_recommendation(disease):
 #   df = create_enhanced_dataset()
  #  return df[df['disease'] == disease]['urgency'].values[0]

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('errors/500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)