import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
from PIL import Image
import pickle
import mysql.connector
import pandas as pd
import datetime
import time
import io
import hashlib

# Page config for mobile responsiveness
st.set_page_config(
    page_title="College Event Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Database connection function
def get_db_connection():
    try:
        if st.secrets.get("database", {}).get("host"):
            return mysql.connector.connect(
                host=st.secrets["database"]["host"],
                user=st.secrets["database"]["user"],
                password=st.secrets["database"]["password"],
                database=st.secrets["database"]["name"],
                port=st.secrets["database"].get("port", 3306)
            )
        else:
            return mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="attendance_system"
            )
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# Event Management Functions
def create_event(event_name, event_date, event_description=""):
    """Create a new event in the database"""
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            query = """
            INSERT INTO events (event_name, event_date, event_description, created_at)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (event_name, event_date, event_description, datetime.datetime.now()))
            conn.commit()
            cursor.close()
            conn.close()
            return True
    except mysql.connector.Error as err:
        st.error(f"Error creating event: {err}")
        return False

def get_all_events():
    """Get all events from database"""
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM events ORDER BY event_date DESC")
            events = cursor.fetchall()
            cursor.close()
            conn.close()
            return events
    except mysql.connector.Error as err:
        st.error(f"Error fetching events: {err}")
        return []

def delete_event(event_id):
    """Delete an event from database"""
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events WHERE id = %s", (event_id,))
            conn.commit()
            cursor.close()
            conn.close()
            return True
    except mysql.connector.Error as err:
        st.error(f"Error deleting event: {err}")
        return False

# Login function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_login(username, password):
    admin_credentials = {
        "admin": hash_password("admin123"),
        "teacher": hash_password("teacher123")
    }
    return admin_credentials.get(username) == hash_password(password)

def login_page():
    # Add system title at the top
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #667eea; font-size: 2.5rem; margin-bottom: 0.5rem;">
            🎓 College Event Attendance System
        </h1>
        <p style="color: #666; font-size: 1.1rem; margin: 0;">
            AI-Powered Face Recognition for Student Attendance
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.title("🔐 Admin Login")
    st.markdown("Please enter your credentials to access the system.")
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="admin")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        submit = st.form_submit_button("Login", use_container_width=True)
        
        if submit:
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Login successful!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid credentials!")
    
    st.markdown("### Default Credentials:")
    st.info("Username: `admin` | Password: `admin123`")
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize face detector
@st.cache_resource
def load_face_detector():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def detect_faces(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        maxSize=(500, 500),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    valid_faces = []
    for (x, y, w, h) in faces:
        aspect_ratio = w / h
        if 0.6 <= aspect_ratio <= 1.5:
            image_area = image.shape[0] * image.shape[1]
            face_area = w * h
            if face_area / image_area > 0.005:
                valid_faces.append((x, y, x+w, y+h))
    return valid_faces

def extract_face_features(image, face_coords):
    x1, y1, x2, y2 = face_coords
    face_roi = image[y1:y2, x1:x2]
    if face_roi.size == 0:
        return None
    face_roi = cv2.resize(face_roi, (100, 100))
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
    features = gray_face.flatten().astype(np.float32)
    features = features / 255.0
    return features

def compare_faces(known_features, unknown_features, threshold=0.5):
    if known_features is None or unknown_features is None:
        return False, 0.0
    dot_product = np.dot(known_features, unknown_features)
    norm_a = np.linalg.norm(known_features)
    norm_b = np.linalg.norm(unknown_features)
    if norm_a == 0 or norm_b == 0:
        return False, 0.0
    similarity = dot_product / (norm_a * norm_b)
    return similarity > threshold, similarity

def save_attendance_record(student_id, student_name, event_name, attendance_type="time_in"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            
            check_query = """
            SELECT id, time_out FROM attendance_records 
            WHERE student_id = %s AND event_name = %s AND date = %s
            ORDER BY timestamp DESC LIMIT 1
            """
            cursor.execute(check_query, (student_id, event_name, date))
            existing = cursor.fetchone()
            
            if existing and attendance_type == "time_out":
                update_query = """
                UPDATE attendance_records 
                SET time_out = %s 
                WHERE id = %s
                """
                cursor.execute(update_query, (timestamp, existing[0]))
            else:
                query = """
                INSERT INTO attendance_records (student_id, student_name, event_name, timestamp, date, time_in, time_out)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                time_in = timestamp if attendance_type == "time_in" else None
                time_out = timestamp if attendance_type == "time_out" else None
                cursor.execute(query, (student_id, student_name, event_name, timestamp, date, time_in, time_out))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
    except mysql.connector.Error as err:
        st.error(f"❌ Error saving attendance: {err}")
        return False

def get_attendance_summary(event_name):
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            query = "SELECT * FROM attendance_records WHERE event_name = %s ORDER BY timestamp DESC"
            cursor.execute(query, (event_name,))
            records = cursor.fetchall()
            cursor.close()
            conn.close()
            return records
    except mysql.connector.Error as err:
        st.error(f"❌ Error fetching attendance: {err}")
        return []

def load_students():
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT student_id, student_name, face_features FROM students")
            students = {}
            for row in cursor.fetchall():
                features = pickle.loads(row['face_features']) if row['face_features'] else None
                students[row['student_id']] = {
                    'name': row['student_name'],
                    'features': features
                }
            cursor.close()
            conn.close()
            return students
    except mysql.connector.Error as err:
        st.error(f"❌ Error loading students: {err}")
        return {}

def check_duplicate_student(student_id, student_name):
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            query = "SELECT student_id, student_name FROM students WHERE student_id = %s OR student_name = %s"
            cursor.execute(query, (student_id, student_name))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return result
    except mysql.connector.Error as err:
        st.error(f"❌ Error checking duplicates: {err}")
        return None

def save_student(student_id, student_name, features):
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            features_blob = pickle.dumps(features)
            query = """
            INSERT INTO students (student_id, student_name, face_features)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE student_name = %s, face_features = %s
            """
            cursor.execute(query, (student_id, student_name, features_blob, student_name, features_blob))
            conn.commit()
            cursor.close()
            conn.close()
            return True
    except mysql.connector.Error as err:
        st.error(f"❌ Error saving student: {err}")
        return False

def search_students(search_term):
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            query = """
            SELECT student_id, student_name FROM students 
            WHERE student_id LIKE %s OR student_name LIKE %s
            """
            search_pattern = f"%{search_term}%"
            cursor.execute(query, (search_pattern, search_pattern))
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            return results
    except mysql.connector.Error as err:
        st.error(f"❌ Error searching students: {err}")
        return []

def delete_student(student_id):
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            query = "DELETE FROM students WHERE student_id = %s"
            cursor.execute(query, (student_id,))
            conn.commit()
            cursor.close()
            conn.close()
            return True
    except mysql.connector.Error as err:
        st.error(f"❌ Error deleting student: {err}")
        return False

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
    st.stop()

# Initialize other session states after login
if "students" not in st.session_state:
    st.session_state.students = load_students()

if "current_event" not in st.session_state:
    st.session_state.current_event = ""

if "current_event_id" not in st.session_state:
    st.session_state.current_event_id = None

if "attendance_today" not in st.session_state:
    st.session_state.attendance_today = []

if "camera_retry_count" not in st.session_state:
    st.session_state.camera_retry_count = 0

# Load face detector
try:
    face_cascade = load_face_detector()
    if face_cascade.empty():
        st.error("❌ Failed to load face detection system")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading face detection: {e}")
    st.stop()

# Header with logout
col_header_left, col_header_right = st.columns([5, 1])

with col_header_left:
    st.markdown(f"""
    <div class="header-container" style="margin-bottom: 0;">
        <h1 class="header-title">🎓 College Event Attendance System</h1>
        <p class="header-subtitle">AI-Powered Face Recognition for Student Attendance</p>
        <p class="header-subtitle">Welcome, {st.session_state.username}!</p>
    </div>
    """, unsafe_allow_html=True)

with col_header_right:
    st.markdown("<div style='padding-top: 1.5rem;'></div>", unsafe_allow_html=True)
    if st.button("🚪 Logout", key="logout_btn", help="Logout from system", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.clear()
        st.rerun()

# Tabs for different functions
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📅 Events", "👥 Student Registration", "✅ Take Attendance", "📊 View Records", "🔍 Manage Students", "⚙️ System Management"])

with tab1:
    st.header("📅 Event Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("➕ Create New Event")
        with st.form("create_event_form"):
            new_event_name = st.text_input("Event Name", placeholder="e.g., Computer Science Seminar 2024")
            new_event_date = st.date_input("Event Date", value=datetime.date.today())
            new_event_desc = st.text_area("Event Description (Optional)", placeholder="Brief description of the event...")
            
            if st.form_submit_button("➕ Create Event", use_container_width=True):
                if new_event_name:
                    if create_event(new_event_name, new_event_date, new_event_desc):
                        st.success(f"✅ Event '{new_event_name}' created successfully!")
                        st.session_state.current_event = new_event_name
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to create event!")
                else:
                    st.error("❌ Please enter an event name!")
    
    with col2:
        st.subheader("📋 Quick Stats")
        all_events = get_all_events()
        st.metric("Total Events", len(all_events))
        st.metric("Active Event", "Yes" if st.session_state.current_event else "No")
    
    st.markdown("---")
    st.subheader("📋 All Events")
    
    all_events = get_all_events()
    if all_events:
        for event in all_events:
            is_active = st.session_state.current_event == event['event_name']
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                active_class = "active" if is_active else ""
                st.markdown(f"""
                <div class="event-list-item {active_class}">
                    <h4 style="margin: 0; color: #333;">📅 {event['event_name']}</h4>
                    <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">
                        📆 {event['event_date']} | Created: {event['created_at'].strftime('%Y-%m-%d %H:%M')}
                    </p>
                    {f"<p style='margin: 0.5rem 0 0 0; color: #666; font-size: 0.85rem;'>{event['event_description']}</p>" if event.get('event_description') else ""}
                    {"<p style='margin: 0.5rem 0 0 0; color: #667eea; font-weight: bold; font-size: 0.9rem;'>✅ Currently Active</p>" if is_active else ""}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("📍 Select", key=f"select_event_{event['id']}", use_container_width=True):
                    st.session_state.current_event = event['event_name']
                    st.session_state.current_event_id = event['id']
                    st.success(f"✅ Event '{event['event_name']}' selected!")
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_event_{event['id']}", use_container_width=True):
                    if delete_event(event['id']):
                        if st.session_state.current_event == event['event_name']:
                            st.session_state.current_event = ""
                            st.session_state.current_event_id = None
                        st.success(f"✅ Event deleted!")
                        st.rerun()
    else:
        st.info("📋 No events created yet. Create your first event above!")
    
    if st.session_state.current_event:
        st.markdown("---")
        st.markdown(f"""
        <div class="event-card">
            <h3 style="margin: 0;">📍 Currently Active Event</h3>
            <h2 style="margin: 0.5rem 0;">{st.session_state.current_event}</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">All attendance operations will be recorded under this event</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.header("👥 Student Registration")
    st.info("Register students using webcam or upload photos")
    
    reg_method = st.radio("Choose registration method:", ["📸 Upload Photo", "📹 Use Webcam"], horizontal=True)
    
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("Student ID", placeholder="e.g., CS-2024-001")
    with col2:
        student_name = st.text_input("Student Name", placeholder="e.g., John Doe")
    
    if student_id and student_name:
        duplicate = check_duplicate_student(student_id, student_name)
        if duplicate:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ <strong>Warning:</strong> Student with ID "{duplicate[0]}" or name "{duplicate[1]}" already exists!
            </div>
            """, unsafe_allow_html=True)
    
    if reg_method == "📸 Upload Photo":
        uploaded_file = st.file_uploader("Upload student photo", type=["jpg", "png", "jpeg"])
        
        if uploaded_file and student_id and student_name:
            if not check_duplicate_student(student_id, student_name):
                try:
                    pil_image = Image.open(uploaded_file).convert("RGB")
                    image_array = np.array(pil_image)
                    
                    faces = detect_faces(image_array, face_cascade)
                    
                    if faces:
                        face_coords = faces[0]
                        features = extract_face_features(image_array, face_coords)
                        
                        if features is not None:
                            if save_student(student_id, student_name, features):
                                st.session_state.students[student_id] = {
                                    "name": student_name,
                                    "features": features
                                }
                                
                                result_image = image_array.copy()
                                x1, y1, x2, y2 = face_coords
                                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                cv2.putText(result_image, f"{student_name}", (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                
                                st.markdown(f"""
                                <div class="success-box">
                                    ✅ <strong>Student registered successfully!</strong><br>
                                    ID: {student_id}<br>
                                    Name: {student_name}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.image(result_image, caption=f"Registered: {student_name}", width=300)
                        else:
                            st.error("❌ Could not extract face features. Try a clearer photo.")
                    else:
                        st.error("❌ No face detected. Please use a clear photo with a visible face.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
            else:
                st.error("❌ Cannot register: Student ID or Name already exists!")
    
    else:  # Webcam registration - keeping original implementation
        if not student_id or not student_name:
            st.warning("⚠️ Please enter Student ID and Name first!")
        elif check_duplicate_student(student_id, student_name):
            st.error("❌ Cannot register: Student ID or Name already exists!")
        else:
            st.info("📹 Look at the camera and click 'Capture & Register' when ready")
            
            if "webcam_key" not in st.session_state:
                st.session_state.webcam_key = 0
            
            class RegistrationProcessor:
                def __init__(self):
                    self.latest_frame = None
                    self.error_count = 0
                    
                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    try:
                        img = frame.to_ndarray(format="bgr24")
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        self.latest_frame = rgb_img.copy()
                        self.error_count = 0
                        
                        faces = detect_faces(rgb_img, face_cascade)
                        
                        for face_coords in faces:
                            x1, y1, x2, y2 = face_coords
                            face_width = x2 - x1
                            face_height = y2 - y1
                            
                            if face_width > 50 and face_height > 50:
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(img, "Face detected - Ready!", (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            else:
                                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
                                cv2.putText(img, "Move closer", (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        
                        if len(faces) == 0:
                            cv2.putText(img, "Please position your face in view", (50, 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        return av.VideoFrame.from_ndarray(img, format="bgr24")
                        
                    except Exception as e:
                        self.error_count += 1
                        if self.error_count > 10:
                            st.session_state.camera_retry_count += 1
                        print(f"Registration processor error: {e}")
                        return frame
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                try:
                    webrtc_ctx = webrtc_streamer(
                        key=f"registration_{st.session_state.webcam_key}",
                        mode=WebRtcMode.SENDRECV,
                        rtc_configuration={
                            "iceServers": [
                                {"urls": ["stun:stun.l.google.com:19302"]},
                                {"urls": ["stun:stun1.l.google.com:19302"]}
                            ]
                        },
                        video_processor_factory=RegistrationProcessor,
                        media_stream_constraints={
                            "video": {
                                "width": {"min": 320, "ideal": 640, "max": 1280},
                                "height": {"min": 240, "ideal": 480, "max": 720},
                                "frameRate": {"min": 5, "ideal": 10, "max": 15}
                            }, 
                            "audio": False
                        },
                        async_processing=True
                    )
                except Exception as e:
                    st.error(f"❌ Camera initialization failed: {e}")
                    webrtc_ctx = None
            
            with col2:
                st.markdown("### 📸 Registration")
                
                if webrtc_ctx and webrtc_ctx.video_processor:
                    st.success("✅ Camera active")
                    
                    if st.button("📸 Capture & Register Student", type="primary", use_container_width=True):
                        if webrtc_ctx.video_processor and hasattr(webrtc_ctx.video_processor, 'latest_frame'):
                            if webrtc_ctx.video_processor.latest_frame is not None:
                                try:
                                    image_array = webrtc_ctx.video_processor.latest_frame
                                    faces = detect_faces(image_array, face_cascade)
                                    
                                    if faces:
                                        face_coords = faces[0]
                                        features = extract_face_features(image_array, face_coords)
                                        
                                        if features is not None and save_student(student_id, student_name, features):
                                            st.session_state.students[student_id] = {
                                                "name": student_name,
                                                "features": features
                                            }
                                            
                                            st.success(f"✅ {student_name} registered!")
                                            
                                            result_image = image_array.copy()
                                            x1, y1, x2, y2 = face_coords
                                            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                            st.image(result_image, caption=f"Captured: {student_name}", width=250)
                                            
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            st.error("❌ Could not extract face features or save student")
                                    else:
                                        st.error("❌ No face detected in capture!")
                                        
                                except Exception as e:
                                    st.error(f"❌ Capture failed: {str(e)}")
                            else:
                                st.warning("⚠️ No frame available - wait a moment")
                        else:
                            st.error("❌ Camera not ready")
                elif st.session_state.camera_retry_count > 3:
                    st.error("❌ Camera appears to be having issues. Please check your connection and refresh the page.")
                else:
                    st.info("📹 Starting camera...")
                
                if st.button("🔄 Reset Camera", key="reset_registration_cam", help="Click if camera is not working"):
                    st.session_state.webcam_key += 1
                    st.session_state.camera_retry_count = 0
                    st.rerun()
    
    if st.session_state.students:
        st.markdown("### 📋 Recently Registered Students")
        recent_students = list(st.session_state.students.items())[-5:]
        for student_id, data in recent_students:
            st.markdown(f"""
            <div class="student-list">
                🎓 <strong>{data['name']}</strong> (ID: {student_id})
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.header("✅ Take Attendance")
    
    if not st.session_state.current_event:
        st.warning("⚠️ Please select an event first from the Events tab!")
    elif not st.session_state.students:
        st.warning("⚠️ Please register students first!")
    else:
        st.markdown(f"""
        <div class="event-card">
            <h4 style="margin: 0;">📅 Taking attendance for:</h4>
            <h3 style="margin: 0.5rem 0 0 0;">{st.session_state.current_event}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        attendance_type = st.radio("Attendance Type:", ["⏰ Time In", "🚪 Time Out"], horizontal=True)
        debug_mode = st.checkbox("🔍 Debug Mode (Show detection details)")
        
        if debug_mode:
            st.info(f"📊 Registered students: {len(st.session_state.students)}")
        
        att_method = st.radio("Choose attendance method:", ["📸 Upload Photo", "📹 Live Webcam"], horizontal=True)
        
        if att_method == "📸 Upload Photo":
            uploaded_attendance = st.file_uploader("Upload photo for attendance", type=["jpg", "png", "jpeg"], key="attendance")
            
            if uploaded_attendance:
                try:
                    pil_image = Image.open(uploaded_attendance).convert("RGB")
                    image_array = np.array(pil_image)
                    
                    faces = detect_faces(image_array, face_cascade)
                    
                    if faces:
                        result_image = image_array.copy()
                        recognized_students = []
                        
                        for face_coords in faces:
                            features = extract_face_features(image_array, face_coords)
                            
                            if features is not None:
                                best_match = None
                                best_similarity = 0
                                
                                for student_id, data in st.session_state.students.items():
                                    is_match, similarity = compare_faces(data["features"], features, threshold=0.4)
                                    if is_match and similarity > best_similarity:
                                        best_similarity = similarity
                                        best_match = (student_id, data["name"])
                                
                                x1, y1, x2, y2 = face_coords
                                if best_match:
                                    student_id, name = best_match
                                    recognized_students.append((student_id, name))
                                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                    cv2.putText(result_image, f"{name}", (x1, y1 - 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                else:
                                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                                    cv2.putText(result_image, "Unknown", (x1, y1 - 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        st.image(result_image, caption="Attendance Recognition Result", width=500)
                        
                        if recognized_students:
                            att_type = "time_in" if attendance_type == "⏰ Time In" else "time_out"
                            st.markdown(f"### ✅ {attendance_type} Marked For:")
                            for student_id, name in recognized_students:
                                if save_attendance_record(student_id, name, st.session_state.current_event, att_type):
                                    st.markdown(f"""
                                    <div class="success-box">
                                        ✅ <strong>{name}</strong> (ID: {student_id}) - {attendance_type}
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.warning("No registered students recognized in the photo.")
                    else:
                        st.warning("No faces detected in the image.")
                        
                except Exception as e:
                    st.error(f"Error processing attendance: {str(e)}")
        
        else:  # Live webcam attendance
            st.info("📹 Look at the camera and click 'Mark Attendance' when ready")
            
            if "attendance_webcam_key" not in st.session_state:
                st.session_state.attendance_webcam_key = 0
            
            if "attendance_session" not in st.session_state:
                st.session_state.attendance_session = []
            
            class AttendanceProcessor:
                def __init__(self):
                    self.latest_frame = None
                    self.error_count = 0
                    
                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    try:
                        img = frame.to_ndarray(format="bgr24")
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        self.latest_frame = rgb_img.copy()
                        self.error_count = 0
                        
                        faces = detect_faces(rgb_img, face_cascade)
                        
                        for face_coords in faces:
                            x1, y1, x2, y2 = face_coords
                            face_width = x2 - x1
                            face_height = y2 - y1
                            
                            if face_width > 50 and face_height > 50:
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(img, "Face detected - Ready!", (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            else:
                                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
                                cv2.putText(img, "Move closer", (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        
                        if len(faces) == 0:
                            cv2.putText(img, "Please position your face in view", (50, 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        return av.VideoFrame.from_ndarray(img, format="bgr24")
                        
                    except Exception as e:
                        self.error_count += 1
                        if self.error_count > 10:
                            st.session_state.camera_retry_count += 1
                        print(f"Attendance processor error: {e}")
                        return frame
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                try:
                    webrtc_ctx = webrtc_streamer(
                        key=f"attendance_{st.session_state.attendance_webcam_key}",
                        mode=WebRtcMode.SENDRECV,
                        rtc_configuration={
                            "iceServers": [
                                {"urls": ["stun:stun.l.google.com:19302"]},
                                {"urls": ["stun:stun1.l.google.com:19302"]}
                            ]
                        },
                        video_processor_factory=AttendanceProcessor,
                        media_stream_constraints={
                            "video": {
                                "width": {"min": 320, "ideal": 640, "max": 1280},
                                "height": {"min": 240, "ideal": 480, "max": 720},
                                "frameRate": {"min": 5, "ideal": 10, "max": 15}
                            }, 
                            "audio": False
                        },
                        async_processing=True
                    )
                except Exception as e:
                    st.error(f"❌ Camera initialization failed: {e}")
                    webrtc_ctx = None
            
            with col2:
                st.markdown(f"### 📸 Mark {attendance_type}")
                
                if webrtc_ctx and webrtc_ctx.video_processor:
                    st.success("✅ Camera active")
                    
                    if st.button(f"✅ Mark {attendance_type}", type="primary", use_container_width=True):
                        if webrtc_ctx.video_processor and hasattr(webrtc_ctx.video_processor, 'latest_frame'):
                            if webrtc_ctx.video_processor.latest_frame is not None:
                                try:
                                    image_array = webrtc_ctx.video_processor.latest_frame
                                    faces = detect_faces(image_array, face_cascade)
                                    
                                    if faces:
                                        face_coords = faces[0]
                                        features = extract_face_features(image_array, face_coords)
                                        
                                        if features is not None:
                                            best_match = None
                                            best_similarity = 0
                                            
                                            for student_id, data in st.session_state.students.items():
                                                is_match, similarity = compare_faces(data["features"], features, threshold=0.3)
                                                if is_match and similarity > best_similarity:
                                                    best_similarity = similarity
                                                    best_match = (student_id, data["name"])
                                            
                                            if best_match and best_similarity > 0.25:
                                                student_id, name = best_match
                                                att_type = "time_in" if attendance_type == "⏰ Time In" else "time_out"
                                                
                                                if save_attendance_record(student_id, name, st.session_state.current_event, att_type):
                                                    current_time = datetime.datetime.now()
                                                    st.session_state.attendance_session.append((student_id, name, current_time.strftime("%H:%M:%S"), attendance_type))
                                                    st.success(f"✅ {attendance_type} marked for {name}!")
                                                    st.balloons()
                                                    
                                                    result_image = image_array.copy()
                                                    x1, y1, x2, y2 = face_coords
                                                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                                    cv2.putText(result_image, f"{name} ({best_similarity:.2f})", (x1, y1 - 10), 
                                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                                    st.image(result_image, caption=f"{attendance_type}: {name}", width=250)
                                                else:
                                                    st.error("❌ Failed to save attendance record")
                                            else:
                                                st.error("❌ Face not recognized! Please register first.")
                                        else:
                                            st.error("❌ Could not extract face features")
                                    else:
                                        st.error("❌ No face detected in capture!")
                                        
                                except Exception as e:
                                    st.error(f"❌ Attendance failed: {str(e)}")
                            else:
                                st.warning("⚠️ No frame available - wait a moment")
                        else:
                            st.error("❌ Camera not ready")
                elif st.session_state.camera_retry_count > 3:
                    st.error("❌ Camera appears to be having issues. Please check your connection and refresh the page.")
                else:
                    st.info("📹 Starting camera...")
                
                if st.button("🔄 Reset Camera", key="reset_attendance_cam", help="Click if camera is not working"):
                    st.session_state.attendance_webcam_key += 1
                    st.session_state.camera_retry_count = 0
                    st.rerun()
                
                st.markdown("### 📋 Recent Attendance")
                if st.session_state.attendance_session:
                    for student_id, name, time, att_type in st.session_state.attendance_session[-5:]:
                        status_color = "#d4edda" if "In" in att_type else "#fff3cd"
                        st.markdown(f"""
                        <div style="background-color: {status_color}; padding: 0.5rem; margin: 0.2rem 0; border-radius: 5px; font-size: 12px; color: #333;">
                            ✅ <strong>{name}</strong> - {att_type}<br>
                            🕐 {time}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No attendance marked yet")
                
                if st.button("🔄 Clear Session", key="clear_attendance_session", help="Clear current session attendance display"):
                    st.session_state.attendance_session = []
                    st.rerun()

with tab4:
    st.header("📊 Attendance Records")
    
    if st.session_state.current_event:
        records = get_attendance_summary(st.session_state.current_event)
        
        if records:
            st.markdown(f"""
            <div class="event-card">
                <h3 style="margin: 0;">📅 {st.session_state.current_event}</h3>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Total Records: {len(records)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            df = pd.DataFrame(records)
            
            for record in records:
                time_in = record.get('time_in', 'N/A')
                time_out = record.get('time_out', 'Not recorded')
                
                st.markdown(f"""
                <div class="student-list">
                    🎓 <strong>{record['student_name']}</strong> (ID: {record['student_id']})<br>
                    📅 Event: {record['event_name']}<br>
                    ⏰ Time In: {time_in}<br>
                    🚪 Time Out: {time_out}
                </div>
                """, unsafe_allow_html=True)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Attendance Report (CSV)",
                data=csv,
                file_name=f"attendance_{st.session_state.current_event}_{datetime.date.today()}.csv",
                mime="text/csv"
            )
        else:
            st.info("No attendance records found for this event.")
    else:
        st.warning("Please select an event to view records.")

with tab5:
    st.header("🔍 Manage Students")
    
    st.subheader("🔍 Search Students")
    search_term = st.text_input("Search by Student ID or Name", placeholder="Enter student ID or name...")
    
    if search_term:
        search_results = search_students(search_term)
        if search_results:
            st.markdown(f"### Search Results ({len(search_results)} found)")
            for student in search_results:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{student['student_name']}** (ID: {student['student_id']})")
                with col2:
                    if st.button("✏️ Edit", key=f"edit_{student['student_id']}"):
                        st.session_state.edit_student = student
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{student['student_id']}"):
                        if delete_student(student['student_id']):
                            st.success(f"✅ Deleted {student['student_name']}")
                            st.session_state.students = load_students()
                            st.rerun()
        else:
            st.info("No students found matching your search.")
    
    if "edit_student" in st.session_state:
        st.subheader("✏️ Edit Student")
        edit_student = st.session_state.edit_student
        
        with st.form("edit_student_form"):
            new_name = st.text_input("Student Name", value=edit_student['student_name'])
            new_id = st.text_input("Student ID", value=edit_student['student_id'], disabled=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("💾 Update Student"):
                    try:
                        conn = get_db_connection()
                        if conn:
                            cursor = conn.cursor()
                            query = "UPDATE students SET student_name = %s WHERE student_id = %s"
                            cursor.execute(query, (new_name, edit_student['student_id']))
                            conn.commit()
                            cursor.close()
                            conn.close()
                            st.success("✅ Student updated successfully!")
                            st.session_state.students = load_students()
                            del st.session_state.edit_student
                            st.rerun()
                    except mysql.connector.Error as err:
                        st.error(f"❌ Error updating student: {err}")
            
            with col2:
                if st.form_submit_button("❌ Cancel"):
                    del st.session_state.edit_student
                    st.rerun()
    
    st.subheader("👥 All Registered Students")
    if st.session_state.students:
        students_per_page = 10
        total_students = len(st.session_state.students)
        total_pages = (total_students - 1) // students_per_page + 1
        
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("⬅️ Previous") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                st.rerun()
        
        with col3:
            st.write(f"Page {st.session_state.current_page} of {total_pages}")
        
        with col5:
            if st.button("Next ➡️") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
                st.rerun()
        
        start_idx = (st.session_state.current_page - 1) * students_per_page
        end_idx = min(start_idx + students_per_page, total_students)
        
        students_list = list(st.session_state.students.items())
        for i in range(start_idx, end_idx):
            student_id, data = students_list[i]
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"""
                <div class="student-list">
                    🎓 <strong>{data['name']}</strong> (ID: {student_id})
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("✏️", key=f"edit_btn_{student_id}", help="Edit student"):
                    st.session_state.edit_student = {'student_id': student_id, 'student_name': data['name']}
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"delete_btn_{student_id}", help="Delete student"):
                    if delete_student(student_id):
                        st.success(f"✅ Deleted {data['name']}")
                        st.session_state.students = load_students()
                        st.rerun()
    else:
        st.info("No students registered yet.")

with tab6:
    st.header("⚙️ System Management")
    
    st.info("ℹ️ Student data is automatically saved to MySQL after each registration and loaded on app startup.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Backup Database"):
            if st.session_state.students:
                backup_data = {
                    'students': {},
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                for student_id, data in st.session_state.students.items():
                    backup_data['students'][student_id] = {
                        'name': data['name'],
                        'features': data['features'].tolist() if data['features'] is not None else None
                    }
                
                import json
                backup_json = json.dumps(backup_data, indent=2)
                
                st.download_button(
                    label="📥 Download Backup File",
                    data=backup_json,
                    file_name=f"attendance_backup_{datetime.date.today()}.json",
                    mime="application/json"
                )
                st.success("✅ Backup ready for download!")
            else:
                st.warning("No student data to backup")
    
    with col2:
        if st.button("📂 Load Student Data (Manual)"):
            try:
                st.session_state.students = load_students()
                st.success("✅ Student data loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading student data: {e}")
    
    with col3:
        if st.button("🗑️ Clear All Data", type="secondary"):
            with st.form("confirm_delete"):
                st.warning("⚠️ This will delete ALL students and attendance records!")
                confirm = st.text_input("Type 'DELETE' to confirm:")
                
                if st.form_submit_button("🗑️ Confirm Delete"):
                    if confirm == "DELETE":
                        try:
                            conn = get_db_connection()
                            if conn:
                                cursor = conn.cursor()
                                cursor.execute("DELETE FROM attendance_records")
                                cursor.execute("DELETE FROM students")
                                conn.commit()
                                cursor.close()
                                conn.close()
                                st.session_state.students = {}
                                st.session_state.attendance_today = []
                                st.success("✅ All data cleared!")
                                st.rerun()
                        except mysql.connector.Error as err:
                            st.error(f"❌ Error clearing data: {err}")
                    else:
                        st.error("Please type 'DELETE' to confirm")
    
    st.markdown("### 📈 System Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Registered Students", len(st.session_state.students))
    
    with col2:
        if st.session_state.current_event:
            records = get_attendance_summary(st.session_state.current_event)
            unique_students = len(set(record['student_id'] for record in records))
            st.metric("✅ Unique Attendees", unique_students)
    
    with col3:
        if st.session_state.current_event:
            records = get_attendance_summary(st.session_state.current_event)
            st.metric("📊 Total Records", len(records))
    
    with col4:
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        st.metric("🕐 Current Time", current_time)
    
    st.markdown("### 🔧 System Health")
    if st.button("🔍 Test Database Connection"):
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM students")
                student_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM attendance_records")
                attendance_count = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                st.success(f"✅ Database connected successfully! {student_count} students, {attendance_count} attendance records")
            except Exception as e:
                st.error(f"❌ Database query failed: {e}")
        else:
            st.error("❌ Failed to connect to database")

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p>🎓 College Event Attendance System - Enhanced Version with Event Management</p>
    <p>Built with Streamlit • OpenCV • MySQL • AI-Powered Recognition</p>
    <p>Current Event: <strong>{st.session_state.current_event or 'Not Set'}</strong> | 
       Logged in as: <strong>{st.session_state.username}</strong></p>
</div>
""", unsafe_allow_html=True)