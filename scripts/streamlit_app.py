import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime

class EthoShiftApp:
    def __init__(self):
        """Initialize Etho Shift AI Ethical Analyser"""
        self.setup_page()
        if 'page' not in st.session_state:
            st.session_state.page = 'welcome'
        if 'analysis_state' not in st.session_state:
            st.session_state.analysis_state = 'start'
    
    def setup_page(self):
        """Configure page settings"""
        st.set_page_config(
            page_title="Etho Shift AI Ethical Analyser",
            page_icon="🔮",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
            <style>
            .main-title {
                font-size: 3rem;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
                padding: 2rem 0;
                background: linear-gradient(to right, #f8f9fa, #e9ecef, #f8f9fa);
                border-radius: 10px;
            }
            .subtitle {
                font-size: 1.5rem;
                color: #2c3e50;
                text-align: center;
                margin-bottom: 2rem;
            }
            .welcome-section {
                padding: 2rem;
                background-color: white;
                border-radius: 10px;
                margin: 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .feature-box {
                padding: 1.5rem;
                background-color: #f8f9fa;
                border-radius: 8px;
                margin: 1rem 0;
                border-left: 4px solid #1f77b4;
            }
            .nav-button {
                text-align: center;
                padding: 1rem;
                margin: 2rem 0;
            }
            .prototype-notice {
                padding: 1rem;
                background-color: #e3f2fd;
                border-left: 4px solid #2196f3;
                border-radius: 4px;
                margin: 1rem 0;
            }
            
            .disclaimer-box {
                padding: 1.5rem;
                background-color: #fff3e0;
                border-left: 4px solid #ff9800;
                border-radius: 4px;
                margin: 1rem 0;
            }
            
            .version-info {
                text-align: center;
                color: #666;
                font-size: 0.9rem;
                margin-top: 2rem;
            }
            
            .prototype-notice h3,
            .disclaimer-box h3 {
                color: #1a237e;
                margin-bottom: 0.5rem;
            }
            
            .feature-box ul ul {
                margin-left: 1.5rem;
                margin-top: 0.5rem;
                margin-bottom: 1rem;
            }
            
            .feature-box ul li strong {
                color: #1f77b4;
            }
            
            .feature-box ul li ul li {
                color: #2c3e50;
                margin: 0.3rem 0;
            }
            
            .action-box {
                padding: 1rem;
                background-color: #e3f2fd;
                border-radius: 4px;
                margin-top: 2rem;
                text-align: center;
                border-left: 4px solid #2196f3;
            }
            
            .action-box p {
                margin: 0;
                font-weight: 500;
                color: #1565c0;
            }
            
            .status-box {
                padding: 1rem;
                background-color: #e8f5e9;
                border-radius: 4px;
                margin: 1rem 0;
                border-left: 4px solid #4caf50;
            }
            
            .configuration-box {
                padding: 1rem;
                background-color: #f3e5f5;
                border-radius: 4px;
                margin: 1rem 0;
                border-left: 4px solid #9c27b0;
            }
            
            .sidebar-info {
                padding: 0.5rem;
                background-color: #f8f9fa;
                border-radius: 4px;
                margin-bottom: 1rem;
            }
            
            .sidebar-note {
                padding: 0.5rem;
                background-color: #fff3e0;
                border-radius: 4px;
                margin: 1rem 0;
                font-size: 0.9rem;
            }
            </style>
        """, unsafe_allow_html=True)
    
    def show_welcome_page(self):
        """Display welcome page content with prototype notice and disclaimer"""
        # Main title
        st.markdown("<h1 class='main-title'>Etho Shift AI Ethical Analyser</h1>", 
                   unsafe_allow_html=True)
        
        st.markdown("<p class='subtitle'>Guiding Ethical AI Decision-Making</p>", 
                   unsafe_allow_html=True)
        
        # Prototype Notice
        st.markdown("""
        <div class='prototype-notice'>
            <h3>🔬 Prototype Demonstration</h3>
            <p>This is a prototype implementation of the Etho Shift AI Ethical Analyser. 
            The current version demonstrates the Adaptive Ethical Prism Framework (AEPF) 
            in a controlled environment with limited models and scenarios.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Introduction
        st.markdown("""
        <div class='welcome-section'>
            <h2>Welcome to Etho Shift</h2>
            <p>
            Etho Shift implements the Adaptive Ethical Prism Framework (AEPF) to ensure 
            AI systems make decisions that are not just efficient, but ethically sound 
            and contextually appropriate.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # AEPF Explanation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='feature-box'>
                <h3>What is AEPF?</h3>
                <p>The Adaptive Ethical Prism Framework is a sophisticated system that:</p>
                <ul>
                    <li>Analyzes decisions through multiple ethical perspectives</li>
                    <li>Adapts to different contexts and scenarios</li>
                    <li>Ensures balanced and fair outcomes</li>
                    <li>Provides transparent decision rationale</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='feature-box'>
                <h3>Current Implementation</h3>
                <p>This prototype demonstrates:</p>
                <ul>
                    <li>HR Candidate Selection via Gradient Boosting</li>
                    <li>Ethical evaluation of selection criteria</li>
                    <li>Bias detection and mitigation</li>
                    <li>Transparent recommendation generation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # How to Use
        st.markdown("""
        <div class='welcome-section'>
            <h2>How to Use Etho Shift</h2>
            <ol>
                <li><strong>Select Model & Scenario:</strong> Use the sidebar to choose the AI model 
                and specific scenario for analysis.</li>
                <li><strong>Configure Parameters:</strong> Adjust analysis parameters to match your 
                requirements.</li>
                <li><strong>Run Analysis:</strong> Click the 'Run Analysis' button to start the 
                ethical evaluation.</li>
                <li><strong>Review Results:</strong> Examine the detailed report with recommendations 
                and ethical considerations.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("""
        <div class='disclaimer-box'>
            <h3>⚠️ Important Disclaimer</h3>
            <p>This prototype is intended for demonstration and research purposes only:</p>
            <ul>
                <li>All recommendations and evaluations are based on sample data and 
                limited scenarios</li>
                <li>Results should not be interpreted as definitive guidance</li>
                <li>Not intended for production use or real-world decision-making 
                without further validation</li>
                <li>The system is under active development and subject to changes</li>
            </ul>
            <p><strong>By proceeding, you acknowledge that this is a prototype system 
            and agree to use the results only for demonstration purposes.</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Version Information
        st.markdown("""
        <div class='version-info'>
            <p>Version: 0.1.0-prototype<br>
            Last Updated: {}</p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)
        
        # Start button
        st.markdown("<div class='nav-button'>", unsafe_allow_html=True)
        if st.button("Start Analysis", use_container_width=True):
            st.session_state.page = 'analysis'
        st.markdown("</div>", unsafe_allow_html=True)
    
    def create_sidebar(self):
        """Create sidebar navigation"""
        st.sidebar.markdown("### Navigation")
        
        # Page selection
        page = st.sidebar.radio(
            "Select Page",
            ["Welcome", "Analysis"],
            index=0 if st.session_state.page == 'welcome' else 1
        )
        
        st.session_state.page = page.lower()
        
        if st.session_state.page == 'analysis':
            self.create_analysis_sidebar()
    
    def create_analysis_sidebar(self):
        """Create analysis configuration sidebar"""
        st.sidebar.markdown("### Analysis Controls")
        
        # Model info (read-only)
        st.sidebar.markdown("""
        <div class='sidebar-info'>
            <p><strong>Model:</strong> Gradient Boosting</p>
            <p><strong>Scenario:</strong> HR Candidate Selection</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Information about configuration
        st.sidebar.markdown("""
        <div class='sidebar-note'>
            <p>This prototype uses pre-configured parameters optimized for 
            candidate evaluation with ethical considerations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Run button
        if st.sidebar.button("Run Analysis", key='run_button'):
            st.session_state.analysis_state = 'running'
            st.experimental_rerun()
    
    def run(self):
        """Run the Etho Shift application"""
        # Create sidebar
        self.create_sidebar()
        
        # Display appropriate page
        if st.session_state.page == 'welcome':
            self.show_welcome_page()
        else:
            self.show_analysis_page()
    
    def show_analysis_page(self):
        """Show analysis page with step-by-step guidance"""
        # Create message placeholder
        message = st.empty()
        
        # Main content area
        if st.session_state.analysis_state == 'start':
            message.markdown("""
            <div class='welcome-section'>
                <h2>Begin Your Analysis</h2>
                <p>Welcome to the Etho Shift analysis dashboard. This prototype demonstrates 
                ethical AI evaluation using the AEPF framework.</p>
                
                <div class='status-box'>
                    <h3>🔍 Current Status</h3>
                    <p>Ready to begin analysis. Click 'Start Analysis' to proceed.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Start button
            if st.button("Start Analysis", use_container_width=True):
                st.session_state.analysis_state = 'configure'
                st.experimental_rerun()
        
        elif st.session_state.analysis_state == 'configure':
            message.markdown("""
            <div class='welcome-section'>
                <h2>Analysis Configuration</h2>
                <p>The Gradient Boosting Model is configured for candidate evaluation 
                with built-in ethical considerations.</p>
                
                <div class='configuration-box'>
                    <h3>Current Configuration</h3>
                    <ul>
                        <li><strong>Model:</strong> Gradient Boosting</li>
                        <li><strong>Scenario:</strong> HR Candidate Selection</li>
                        <li><strong>Framework:</strong> AEPF v1.0</li>
                    </ul>
                </div>
                
                <div class='feature-box'>
                    <h3>Model Configuration</h3>
                    <ul>
                        <li><strong>Evaluation Metrics:</strong>
                            <ul>
                                <li>Engagement Survey Scores</li>
                                <li>Employee Satisfaction Ratings</li>
                                <li>Special Projects Participation</li>
                                <li>Attendance and Reliability</li>
                            </ul>
                        </li>
                        <li><strong>Bias Mitigation:</strong>
                            <ul>
                                <li>Gender and Race Bias Detection</li>
                                <li>Demographic Balance Monitoring</li>
                                <li>Equal Opportunity Verification</li>
                            </ul>
                        </li>
                    </ul>
                </div>
                
                <div class='action-box'>
                    <p>Click 'Run Analysis' in the sidebar when ready to proceed.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add sidebar options
            self.create_analysis_sidebar()
            
        elif st.session_state.analysis_state == 'running':
            self.show_comprehensive_analysis()
    
    # Your existing analysis methods remain the same...

def main():
    """Run the Etho Shift application"""
    app = EthoShiftApp()
    app.run()

if __name__ == "__main__":
    main()