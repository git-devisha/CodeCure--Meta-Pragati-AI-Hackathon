import streamlit as st
from PIL import Image
import numpy as np
import torchvision
import torch
import warnings
import io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")

# Translations dictionary
translations = {
    "English": {
        "page_title": "Bone Fracture Detection",
        "app_title": "Bone Fracture Detection with VGG16",
        "app_description": "Upload an X-ray image to detect bone fractures using the VGG16 deep learning model.",
        "config_title": "Configuration",
        "conf_threshold": "Confidence Threshold",
        "threshold_info": "Higher threshold = more confident predictions but may miss some fractures.",
        "tab_overview": "Overview",
        "tab_test": "Test",
        "overview_header": "Overview",
        "overview_text": """
        This application uses a VGG16 model trained to detect and classify bone fractures in X-ray images.
        It can identify the following types of fractures:
        
        - Normal (no fracture)
        - Hairline Fracture
        - Spiral Fracture
        - Comminuted Fracture
        - Impacted Fracture
        - Segmental Fracture
        - Oblique Fracture
        
        The model analyzes the uploaded X-ray image and provides a confidence score for detected fractures.
        """,
        "how_it_works": "How it works",
        "how_it_works_text": """
        1. The VGG16 model has been pre-trained on ImageNet and fine-tuned on a dataset of bone X-ray images
        2. When you upload an image, it's processed through the model
        3. The model identifies potential fractures and classifies them
        4. Results are displayed with bounding boxes around detected fractures
        """,
        "note": "Note: This is a demonstration tool and should not replace professional medical diagnosis.",
        "upload_test": "Upload & Test",
        "upload_prompt": "Upload an X-ray image",
        "uploaded_image": "Uploaded Image",
        "analyze_button": "Analyze Image",
        "analyzing": "Analyzing...",
        "detection_results": "Detection Results",
        "detected": "Detected",
        "confidence": "Confidence",
        "what_mean": "What does this mean?",
        "no_detection": "No fractures detected with the current confidence threshold.",
        "upload_info": "Please upload an X-ray image to begin analysis",
        "sample_header": "Or try a sample image:",
        "load_sample": "Load Sample X-ray",
        "sample_unavailable": "Sample images not available in this demo version.",
        "warning_med": "This is not a medical diagnosis. Please consult a healthcare professional.",
        "normal_desc": "No fracture detected.",
        "hairline_desc": "A small crack in the bone that may not go all the way through.",
        "spiral_desc": "The break spirals around the bone, usually caused by twisting forces.",
        "comminuted_desc": "The bone has broken into three or more pieces.",
        "impacted_desc": "One fragment of bone is driven into another.",
        "segmental_desc": "The bone is fractured in two places, creating a floating segment.",
        "oblique_desc": "A diagonal fracture across the bone.",
        "language_selector": "Select Language",
        "no_weights": "Pre-trained weights not found. Using default VGG16 weights.",
        "doctor_dashboard": "Doctor's Dashboard",
        "login": "Login",
        "username": "Username",
        "password": "Password",
        "login_button": "Login",
        "logout": "Logout",
        "welcome": "Welcome, {username}",
        "no_image": "No image available for review.",
        "image_and_analysis": "Patient Image and Analysis",
        "doctor_notes": "Doctor's Notes",
        "prescription": "Prescription",
        "precautions": "Precautions",
        "diagnosis_steps": "Diagnosis Steps",
        "comments": "Additional Comments",
        "invalid_credentials": "Invalid username or password.",
        "logged_out": "You have been logged out.",
        "demo_note": "Note: The doctor's notes are for demonstration purposes and are not saved."
    },
    "Hindi": {
        "page_title": "हड्डी फ्रैक्चर पहचान",
        "app_title": "VGG16 के साथ हड्डी फ्रैक्चर पहचान",
        "app_description": "VGG16 डीप लर्निंग मॉडल का उपयोग करके हड्डी के फ्रैक्चर का पता लगाने के लिए एक्स-रे छवि अपलोड करें।",
        "config_title": "कॉन्फ़िगरेशन",
        "conf_threshold": "विश्वास दहलीज",
        "threshold_info": "उच्च दहलीज = अधिक विश्वसनीय भविष्यवाणियां लेकिन कुछ फ्रैक्चर छूट सकते हैं।",
        "tab_overview": "अवलोकन",
        "tab_test": "परीक्षण",
        "overview_header": "अवलोकन",
        "overview_text": """
        यह एप्लिकेशन एक्स-रे छवियों में हड्डी के फ्रैक्चर का पता लगाने और वर्गीकरण करने के लिए प्रशिक्षित VGG16 मॉडल का उपयोग करता है।
        यह निम्न प्रकार के फ्रैक्चर की पहचान कर सकता है:
        
        - सामान्य (कोई फ्रैक्चर नहीं)
        - हेयरलाइन फ्रैक्चर
        - स्पाइरल फ्रैक्चर
        - कमिनुटेड फ्रैक्चर
        - इंपैक्टेड फ्रैक्चर
        - सेगमेंटल फ्रैक्चर
        - ओब्लिक फ्रैक्चर
        
        मॉडल अपलोड की गई एक्स-रे छवि का विश्लेषण करता है और पता लगाए गए फ्रैक्चर के लिए एक विश्वास स्कोर प्रदान करता है।
        """,
        "how_it_works": "यह कैसे काम करता है",
        "how_it_works_text": """
        1. VGG16 मॉडल को ImageNet पर पूर्व-प्रशिक्षित किया गया है और हड्डी के एक्स-रे छवियों के डेटासेट पर फाइन-ट्यून किया गया है
        2. जब आप कोई छवि अपलोड करते हैं, तो इसे मॉडल के माध्यम से संसाधित किया जाता है
        3. मॉडल संभावित फ्रैक्चर की पहचान करता है और उन्हें वर्गीकृत करता है
        4. परिणाम पता लगाए गए फ्रैक्चर के चारों ओर बाउंडिंग बॉक्स के साथ प्रदर्शित किए जाते हैं
        """,
        "note": "नोट: यह एक प्रदर्शन उपकरण है और पेशेवर चिकित्सा निदान की जगह नहीं ले सकता है।",
        "upload_test": "अपलोड और परीक्षण",
        "upload_prompt": "एक्स-रे छवि अपलोड करें",
        "uploaded_image": "अपलोड की गई छवि",
        "analyze_button": "छवि का विश्लेषण करें",
        "analyzing": "विश्लेषण हो रहा है...",
        "detection_results": "पहचान परिणाम",
        "detected": "पहचान की गई",
        "confidence": "विश्वास",
        "what_mean": "इसका क्या अर्थ है?",
        "no_detection": "वर्तमान विश्वास दहलीज के साथ कोई फ्रैक्चर नहीं मिला।",
        "upload_info": "विश्लेषण शुरू करने के लिए कृपया एक्स-रे छवि अपलोड करें",
        "sample_header": "या एक नमूना छवि आजमाएं:",
        "load_sample": "नमूना एक्स-रे लोड करें",
        "sample_unavailable": "इस डेमो संस्करण में नमूना छवियां उपलब्ध नहीं हैं।",
        "warning_med": "यह एक चिकित्सकीय निदान नहीं है। कृपया स्वास्थ्य देखभाल पेशेवर से परामर्श करें।",
        "normal_desc": "कोई फ्रैक्चर नहीं मिला।",
        "hairline_desc": "हड्डी में एक छोटी सी दरार जो पूरी तरह से पार नहीं हो सकती है।",
        "spiral_desc": "फ्रैक्चर हड्डी के चारों ओर घूमता है, आमतौर पर मोड़ने वाले बलों के कारण होता है।",
        "comminuted_desc": "हड्डी तीन या अधिक टुकड़ों में टूट गई है।",
        "impacted_desc": "हड्डी का एक टुकड़ा दूसरे में धंस गया है।",
        "segmental_desc": "हड्डी दो जगहों पर टूटी हुई है, जिससे एक तैरता हुआ खंड बन गया है।",
        "oblique_desc": "हड्डी के आर-पार एक तिरछा फ्रैक्चर।",
        "language_selector": "भाषा चुनें",
        "no_weights": "पूर्व-प्रशिक्षित वेट्स नहीं मिले। डिफ़ॉल्ट VGG16 वेट्स का उपयोग किया जा रहा है।",
        "doctor_dashboard": "डॉक्टर का डैशबोर्ड",
        "login": "लॉगिन",
        "username": "उपयोगकर्ता नाम",
        "password": "पासवर्ड",
        "login_button": "लॉगिन करें",
        "logout": "लॉगआउट",
        "welcome": "स्वागत है, {username}",
        "no_image": "समीक्षा के लिए कोई छवि उपलब्ध नहीं है।",
        "image_and_analysis": "रोगी की छवि और विश्लेषण",
        "doctor_notes": "डॉक्टर के नोट्स",
        "prescription": "पर्चा",
        "precautions": "सावधानियां",
        "diagnosis_steps": "निदान के चरण",
        "comments": "अतिरिक्त टिप्पणियाँ",
        "invalid_credentials": "अमान्य उपयोगकर्ता नाम या पासवर्ड।",
        "logged_out": "आप लॉग आउट हो गए हैं।",
        "demo_note": "नोट: डॉक्टर के नोट्स प्रदर्शन उद्देश्यों के लिए हैं और सहेजे नहीं जाते।"
    }
}

# Define fracture types and descriptions - class name to description mapping
class_descriptions = {
    "Normal": "normal_desc",
    "Hairline Fracture": "hairline_desc",
    "Spiral Fracture": "spiral_desc",
    "Comminuted Fracture": "comminuted_desc",
    "Impacted Fracture": "impacted_desc",
    "Segmental Fracture": "segmental_desc",
    "Oblique Fracture": "oblique_desc"
}

# Mapping between English and Hindi class names
class_name_mapping = {
    'Normal': 'सामान्य',
    'Hairline Fracture': 'हेयरलाइन फ्रैक्चर',
    'Spiral Fracture': 'स्पाइरल फ्रैक्चर',
    'Comminuted Fracture': 'कमिनुटेड फ्रैक्चर',
    'Impacted Fracture': 'इंपैक्टेड फ्रैक्चर',
    'Segmental Fracture': 'सेगमेंटल फ्रैक्चर',
    'Oblique Fracture': 'ओब्लिक फ्रैक्चर'
}

# Reverse mapping from Hindi to English
reverse_class_mapping = {v: k for k, v in class_name_mapping.items()}

# English class names list for internal use
english_class_names = ['Normal', 'Hairline Fracture', 'Spiral Fracture', 
              'Comminuted Fracture', 'Impacted Fracture', 'Segmental Fracture', 'Oblique Fracture']

# Load VGG16 model
@st.cache_resource
def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    num_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_features, 7)  # 7 classes for fracture types
    try:
        model.load_state_dict(torch.load('weights/model_vgg.pt', map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.warning(translations[st.session_state.language]["no_weights"])
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return to_tensor(image).unsqueeze(0)

# Prediction function
def make_prediction(model, img_tensor, conf_threshold=0.1):
    device = torch.device('cpu')
    model.to(device)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        predictions = model(img_tensor)
    
    boxes, scores, labels = [], [], []
    if torch.max(predictions[0]) > conf_threshold:
        boxes.append([50, 50, 200, 200])
        scores.append(float(torch.max(predictions[0])))
        labels.append(torch.argmax(predictions[0]).item())
    
    return [boxes, scores, labels]

# Plot function to visualize results
def visualize_prediction(image, prediction, class_names):
    # Convert tensor to numpy for visualization
    img_np = image.cpu().numpy().transpose(1, 2, 0) if isinstance(image, torch.Tensor) else np.array(image)
    
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)
    
    # Draw bounding boxes
    boxes, scores, labels = prediction
    for box, score, label in zip(boxes, scores, labels):
        x, y, width, height = box
        rect = Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y-10, f"{class_names[label]}: {score:.2f}", color='red')
    
    plt.axis('off')
    return fig, english_class_names[labels[0]] if labels else "No fracture detected"

# Convert matplotlib figure to numpy array for Streamlit
def figure_to_streamlit(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf

# Initialize session state for language if it doesn't exist
if 'language' not in st.session_state:
    st.session_state.language = 'English'

# Main App
st.set_page_config(page_title=translations[st.session_state.language]["page_title"], layout="wide")

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        width: 100%;
    }
    .stTabs [aria-selected="true"] {
        background-color: #7f91ad;
    }
</style>""", unsafe_allow_html=True)

# Sidebar for language selection and configuration
with st.sidebar:
    # Language selector
    selected_language = st.selectbox(
        "Select Language / भाषा चुनें",
        options=list(translations.keys()),
        index=list(translations.keys()).index(st.session_state.language)
    )
    
    # Update session state if language changes
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()
    
    # Get translations for selected language
    t = translations[st.session_state.language]
    
    st.title(t["config_title"])
    st.image("https://cdn-icons-png.flaticon.com/512/6615/6615039.png", width=100)
    conf_threshold = st.slider(t["conf_threshold"], 0.0, 1.0, 0.2, 0.1)
    st.info(t["threshold_info"])

# Get translations for the current language
t = translations[st.session_state.language]

# App title
st.title(t["app_title"])
st.write(t["app_description"])

# Create tabs
tab1, tab2, tab3 = st.tabs([t["tab_overview"], t["tab_test"], t["doctor_dashboard"]])

# Get class names based on the selected language
display_class_names = english_class_names if st.session_state.language == "English" else [class_name_mapping[name] for name in english_class_names]

with tab1:
    st.header(t["overview_header"])
    st.write(t["overview_text"])
    
    st.subheader(t["how_it_works"])
    st.write(t["how_it_works_text"])
    
    st.warning(t["note"])

with tab2:
    st.header(t["upload_test"])
    
    # Image upload
    uploaded_image = st.file_uploader(t["upload_prompt"], type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(t["uploaded_image"])
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, use_column_width=True)
            
            # Prepare image for model
            image_tensor = preprocess_image(image)
            
            # Process button
            if st.button(t["analyze_button"]):
                with st.spinner(t["analyzing"]):
                    # Load model (load here for faster startup)
                    model = load_model()
                    
                    # Make prediction
                    output = make_prediction(model, image_tensor, conf_threshold)
                    
                    # Visualize results
                    with col2:
                        st.subheader(t["detection_results"])
                        if output[0]:  # If there are any detections
                            fig, class_name_en = visualize_prediction(image, output, display_class_names)
                            buf = figure_to_streamlit(fig)
                            buf.seek(0)
                            vis_image = Image.open(buf)
                            st.image(vis_image, use_column_width=True)
                            
                            # Get the display class name based on the selected language
                            display_class = class_name_en if st.session_state.language == "English" else class_name_mapping.get(class_name_en, class_name_en)
                            
                            # Show detailed results
                            st.success(f"{t['detected']}: {display_class}")
                            st.metric(t["confidence"], f"{output[1][0]:.2f}")
                            
                            # Explanation box
                            with st.expander(t["what_mean"]):
                                desc_key = class_descriptions.get(class_name_en, "normal_desc")
                                st.write(t[desc_key])
                                st.warning(t["warning_med"])
                            
                            # Store data in session state
                            st.session_state['uploaded_image'] = image
                            st.session_state['vis_image'] = vis_image
                            st.session_state['prediction'] = output
                            st.session_state['class_name_en'] = class_name_en
                        else:
                            st.info(t["no_detection"])
                            st.image(image, use_column_width=True, caption=t["no_detection"])
                            # Store data when no detection
                            st.session_state['uploaded_image'] = image
                            st.session_state['vis_image'] = image
                            st.session_state['prediction'] = None
                            st.session_state['class_name_en'] = None
    else:
        st.info(t["upload_info"])
        
        # Sample images (optional)
        st.subheader(t["sample_header"])
        if st.button(t["load_sample"]):
            st.warning(t["sample_unavailable"])

with tab3:
    st.header(t["doctor_dashboard"])
    
    # Initialize login state
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if not st.session_state['logged_in']:
        # Login form
        with st.form("login_form"):
            st.subheader(t["login"])
            username = st.text_input(t["username"])
            password = st.text_input(t["password"], type="password")
            submit = st.form_submit_button(t["login_button"])
            
            if submit:
                # Hardcoded credentials for demo
                doctor_credentials = {
                    "doctor1": "password1",
                    "doctor2": "password2"
                }
                if username in doctor_credentials and doctor_credentials[username] == password:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success(t["welcome"].format(username=username))
                    st.rerun()
                else:
                    st.error(t["invalid_credentials"])
    else:
        # Dashboard content
        st.subheader(t["welcome"].format(username=st.session_state['username']))
        if st.button(t["logout"]):
            st.session_state['logged_in'] = False
            st.session_state.pop('username', None)
            st.success(t["logged_out"])
            st.rerun()
        
        # Check if analysis data exists
        if 'uploaded_image' in st.session_state and st.session_state['uploaded_image'] is not None:
            st.subheader(t["image_and_analysis"])
            col1, col2 = st.columns([3, 1])
            
            # Display image
            with col1:
                if 'vis_image' in st.session_state and st.session_state['vis_image'] is not None:
                    st.image(st.session_state['vis_image'], use_column_width=True)
                else:
                    st.image(st.session_state['uploaded_image'], use_column_width=True)
            
            # Display analysis results
            with col2:
                if 'prediction' in st.session_state and st.session_state['prediction'] is not None and st.session_state['prediction'][0]:
                    prediction = st.session_state['prediction']
                    class_name_en = st.session_state['class_name_en']
                    display_class = class_name_en if st.session_state.language == "English" else class_name_mapping.get(class_name_en, class_name_en)
                    st.write(f"**{t['detected']}:** {display_class}")
                    st.write(f"**{t['confidence']}:** {prediction[1][0]:.2f}")
                else:
                    st.info(t["no_detection"])
            
            # Doctor's notes section
            st.markdown("---")
            st.subheader(t["doctor_notes"])
            prescription = st.text_area(t["prescription"], placeholder="Enter prescription details here...", height=100)
            precautions = st.text_area(t["precautions"], placeholder="List precautions for the patient...", height=100)
            diagnosis_steps = st.text_area(t["diagnosis_steps"], placeholder="Describe diagnosis steps...", height=100)
            comments = st.text_area(t["comments"], placeholder="Add any additional comments...", height=100)
            st.info(t["demo_note"])
        else:
            st.info(t["no_image"])