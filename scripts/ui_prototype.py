import streamlit as st
import time
import numpy as np
import pandas as pd
from scripts.models.gradient_boost.test_gradient_boost import test_gradient_boost_model
import matplotlib.pyplot as plt

def format_candidate_data(report_content):
    """Convert the candidate data into a pandas DataFrame for better display"""
    # Extract the candidates section
    start_idx = report_content.find("RECOMMENDED CANDIDATES")
    end_idx = report_content.find("INTERVIEW RECOMMENDATIONS")
    candidates_section = report_content[start_idx:end_idx]
    
    # Parse the data into a list of dictionaries
    lines = candidates_section.split('\n')
    data = []
    for line in lines[3:-1]:  # Skip header lines and empty last line
        if line.strip() and not line.startswith('-'):
            parts = line.split('|')
            if len(parts) == 8:  # Ensure we have all columns
                data.append({
                    'Rank': int(parts[0].strip()),
                    'ID': parts[1].strip(),
                    'Department': parts[2].strip(),
                    'Position': parts[3].strip(),
                    'Performance': parts[4].strip(),
                    'Engagement': float(parts[5].strip()),
                    'Satisfaction': float(parts[6].strip()),
                    'Key Strength': parts[7].strip()
                })
    
    return pd.DataFrame(data)

def format_strength(strength):
    """Convert long strength descriptions to concise versions"""
    mapping = {
        'High Engagement': 'Engaged',
        'Project Leadership': 'Leader',
        'Outstanding Performance': 'Top Perf',
        'Strong Reliability': 'Reliable',
        'Balanced Performance': 'Balanced',
        'High Satisfaction': 'Satisfied'
    }
    return mapping.get(strength, strength)

def format_position(position):
    """Shorten position titles"""
    mapping = {
        'Production Tech': 'Prod Tech',
        'Data Analyst': 'Data Anlst',
        'Network Engineer': 'Net Eng',
        'Sales Manager': 'Sales Mgr',
        'Software Engineer': 'SW Eng',
        'Area Sales Manager': 'Area Mgr',
        'Database Administrator': 'DB Admin',
        'Sr. Network Engineer': 'Sr Net Eng',
        'IT Support': 'IT Supp',
        'Production Manager': 'Prod Mgr'
    }
    return mapping.get(position, position[:8])

def format_department(dept):
    """Shorten department names"""
    mapping = {
        'Software Engineering': 'SW Eng',
        'Production': 'Prod',
        'Information Technology': 'IT',
        'Human Resources': 'HR'
    }
    return mapping.get(dept, dept)

def show_welcome():
    """Display the welcome page with simplified styling"""
    st.title("Etho Shift AI Ethical Analyser")

    st.markdown("""
    Welcome to Etho Shift! This platform is designed to evaluate and guide ethical decision-making 
    in AI models using the **Adaptive Ethical Prism Framework (AEPF)**.
    """)

    st.header("What is AEPF?")
    st.markdown("""
    The Adaptive Ethical Prism Framework is an ethical assessment system that enables AI to analyze 
    and adapt decisions through different ethical lenses. Each lens focuses on a unique set of values 
    to ensure balanced, context-driven outcomes in automated decisions.
    """)

    st.header("Available Models")
    st.markdown("""
    This prototype includes:
    * **Gradient Boosting Model**: Optimized for HR candidate selection
    * **Random Forest Classifier**: For risk assessment scenarios
    * **Neural Network**: For complex pattern recognition
    """)

    st.subheader("Key Scenarios")
    st.markdown("""
    * HR Candidate Selection
    * Credit Risk Assessment
    * Medical Diagnosis Support
    """)

    st.warning("""
    ⚠️ **Prototype Notice**
    
    This is a prototype demonstration of the Etho Shift AI Ethical Analyser. The current features 
    showcase the AEPF in a controlled environment with limited models and scenarios.
    """)

    st.info("""
    **Disclaimer**
    
    This prototype is intended for demonstration purposes only. The recommendations and evaluations 
    are based on sample data and limited scenarios. **The results should not be interpreted as 
    definitive guidance** and should not be used in real-world decision-making without further validation.
    """)

def show_analysis():
    """Display the analysis page with improved formatting"""
    st.title("Analysis Controls")
    st.write("Please select a model to begin.")
    
    model = st.sidebar.selectbox(
        "Select AI Model",
        ["", "Gradient Boosting Model"],
        index=0,
        help="Choose an AI model to analyze"
    )
    
    if model:
        st.write(f"Model selected: **{model}**")
        
        scenario = st.sidebar.selectbox(
            "Select Scenario",
            ["", "HR Candidate Selection"],
            index=0,
            help="Choose a scenario to analyze"
        )
        
        if scenario:
            st.write(f"Scenario selected: **{scenario}**")
            
            if st.button("Start Analysis", type="primary"):
                with st.spinner("Analyzing candidate pool..."):
                    success, report_path = test_gradient_boost_model()
                    
                    if success:
                        with open(report_path, 'r') as f:
                            report_content = f.read()
                        
                        st.success("Analysis Complete!")
                        
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "Top Candidates", 
                            "Model Insights",
                            "Interview Guide",
                            "Detailed Report"
                        ])
                        
                        with tab1:
                            st.subheader("Top 15 Recommended Candidates")
                            
                            # Convert candidate data to DataFrame
                            df = format_candidate_data(report_content)
                            
                            # Prepare data with condensed text
                            df['Key Strength'] = df['Key Strength'].apply(format_strength)
                            df['Position'] = df['Position'].apply(format_position)
                            df['Department'] = df['Department'].apply(format_department)
                            
                            # Custom CSS for compact table
                            st.markdown("""
                                <style>
                                    .stDataFrame {
                                        font-size: 12px;
                                        padding: 0px;
                                    }
                                    .stDataFrame td {
                                        padding: 3px;
                                        line-height: 1;
                                    }
                                    .stDataFrame th {
                                        padding: 3px;
                                        line-height: 1;
                                    }
                                </style>
                            """, unsafe_allow_html=True)
                            
                            # Display candidate table with minimal spacing
                            st.dataframe(
                                df,
                                column_config={
                                    'Rank': st.column_config.NumberColumn(
                                        '#',
                                        format='%d',
                                        width=35,
                                        help="Rank"
                                    ),
                                    'ID': st.column_config.TextColumn(
                                        'ID',
                                        width=60,
                                        help="Employee ID"
                                    ),
                                    'Department': st.column_config.TextColumn(
                                        'Dept',
                                        width=65,
                                        help="Department"
                                    ),
                                    'Position': st.column_config.TextColumn(
                                        'Pos',
                                        width=75,
                                        help="Position"
                                    ),
                                    'Performance': st.column_config.TextColumn(
                                        'Perf',
                                        width=70,
                                        help="Performance"
                                    ),
                                    'Engagement': st.column_config.NumberColumn(
                                        'Eng',
                                        format='%.0f',  # Remove decimals
                                        width=45,
                                        help="Engagement"
                                    ),
                                    'Satisfaction': st.column_config.NumberColumn(
                                        'Sat',
                                        format='%.0f',  # Remove decimals
                                        width=45,
                                        help="Satisfaction"
                                    ),
                                    'Key Strength': st.column_config.TextColumn(
                                        'Strength',
                                        width=70,
                                        help="Key Strength"
                                    )
                                },
                                hide_index=True,
                                use_container_width=True,
                                height=400  # Reduced height
                            )
                            
                            # Compact legend
                            st.caption("""
                            **Key:** Eng=Engagement(1-5) | Sat=Satisfaction(1-5) | Perf=Performance
                            """)
                            
                            # Display insights below the table
                            st.subheader("Key Insights")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Department Distribution:**")
                                dept_counts = df['Department'].value_counts()
                                for dept, count in dept_counts.items():
                                    st.write(f"- {dept}: {count} candidates ({count/len(df)*100:.0f}%)")
                            
                            with col2:
                                st.write("**Performance Metrics:**")
                                st.write(f"- Exceeds Expectations: {len(df[df['Performance'] == 'Exceeds'])} candidates")
                                st.write(f"- High Engagement (≥4.0): {len(df[df['Engagement'] >= 4.0])} candidates")
                                st.write(f"- High Satisfaction (≥4.0): {len(df[df['Satisfaction'] >= 4.0])} candidates")
                                st.write(f"- Project Leaders: {len(df[df['Key Strength'] == 'Project Leadership'])} candidates")
                        
                        with tab2:
                            st.subheader("Model Performance Summary")
                            start_idx = report_content.find("MODEL SUMMARY")
                            end_idx = report_content.find("EXECUTIVE SUMMARY")
                            if start_idx > -1 and end_idx > -1:
                                st.info(report_content[start_idx:end_idx])
                        
                        with tab3:
                            st.subheader("Interview Focus Areas")
                            start_idx = report_content.find("INTERVIEW RECOMMENDATIONS")
                            end_idx = report_content.find("CLOSING NOTES")
                            if start_idx > -1 and end_idx > -1:
                                recommendations = report_content[start_idx:end_idx]
                                for section in recommendations.split('\n\n'):
                                    if section.strip() and ':' in section:
                                        title, content = section.split(':', 1)
                                        with st.expander(title.strip()):
                                            st.write(content.strip())
                        
                        with tab4:
                            st.subheader("Comprehensive Analysis Report")
                            
                            # Performance Metrics Section
                            st.write("### Performance Distribution")
                            perf_metrics = {
                                'Department Analysis': {
                                    'IT/IS': len(df[df['Department'] == 'IT/IS']),
                                    'Production': len(df[df['Department'] == 'Production']),
                                    'Sales': len(df[df['Department'] == 'Sales']),
                                    'Software Engineering': len(df[df['Department'] == 'Software Engineering'])
                                },
                                'Performance Levels': {
                                    'Exceeds': len(df[df['Performance'] == 'Exceeds']),
                                    'Fully Meets': len(df[df['Performance'] == 'Fully Meets']),
                                    'Needs Improvement': len(df[df['Performance'] == 'Needs Improvement'])
                                },
                                'Engagement Levels': {
                                    'High (≥4.5)': len(df[df['Engagement'] >= 4.5]),
                                    'Medium (3.5-4.4)': len(df[(df['Engagement'] >= 3.5) & (df['Engagement'] < 4.5)]),
                                    'Lower (<3.5)': len(df[df['Engagement'] < 3.5])
                                }
                            }
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("#### Departmental Breakdown")
                                for dept, count in perf_metrics['Department Analysis'].items():
                                    st.write(f"- **{dept}:** {count} candidates ({count/len(df)*100:.1f}%)")
                                
                                st.write("\n#### Performance Ratings")
                                for level, count in perf_metrics['Performance Levels'].items():
                                    st.write(f"- **{level}:** {count} candidates ({count/len(df)*100:.1f}%)")
                            
                            with col2:
                                st.write("#### Engagement Analysis")
                                for level, count in perf_metrics['Engagement Levels'].items():
                                    st.write(f"- **{level}:** {count} candidates ({count/len(df)*100:.1f}%)")
                                
                                st.write("\n#### Key Strengths Distribution")
                                strengths = df['Key Strength'].value_counts()
                                for strength, count in strengths.items():
                                    st.write(f"- **{strength}:** {count} candidates ({count/len(df)*100:.1f}%)")
                            
                            # Detailed Metrics
                            st.write("### Detailed Metrics")
                            metrics_df = pd.DataFrame({
                                'Metric': [
                                    'Average Engagement Score',
                                    'Average Satisfaction Score',
                                    'High Performers Ratio',
                                    'Project Leadership Ratio',
                                    'High Engagement Ratio',
                                    'Department Diversity Score'
                                ],
                                'Value': [
                                    f"{df['Engagement'].mean():.2f}",
                                    f"{df['Satisfaction'].mean():.2f}",
                                    f"{len(df[df['Performance'] == 'Exceeds'])/len(df)*100:.1f}%",
                                    f"{len(df[df['Key Strength'] == 'Project Leadership'])/len(df)*100:.1f}%",
                                    f"{len(df[df['Engagement'] >= 4.0])/len(df)*100:.1f}%",
                                    f"{len(df['Department'].unique())}"
                                ],
                                'Context': [
                                    'Scale: 1-5, Target: ≥4.0',
                                    'Scale: 1-5, Target: ≥4.0',
                                    'Percentage of candidates exceeding expectations',
                                    'Percentage showing leadership qualities',
                                    'Percentage with high engagement scores',
                                    'Number of unique departments represented'
                                ]
                            })
                            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
                            
                            # Risk and Opportunity Analysis
                            st.write("### Risk and Opportunity Analysis")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("#### Potential Risks")
                                risks = [
                                    f"- {len(df[df['Engagement'] < 3.5])} candidates with lower engagement scores",
                                    f"- {len(df[df['Satisfaction'] < 3.5])} candidates with lower satisfaction scores",
                                    f"- {len(df[df['Performance'] == 'Needs Improvement'])} candidates needing performance improvement"
                                ]
                                for risk in risks:
                                    st.write(risk)
                            
                            with col2:
                                st.write("#### Key Opportunities")
                                opportunities = [
                                    f"- {len(df[df['Engagement'] >= 4.5])} candidates showing exceptional engagement",
                                    f"- {len(df[df['Key Strength'] == 'Project Leadership'])} candidates with leadership potential",
                                    f"- {len(df[df['Performance'] == 'Exceeds'])} high-performing candidates"
                                ]
                                for opp in opportunities:
                                    st.write(opp)
                            
                            # Recommendations Summary
                            st.write("### Strategic Recommendations")
                            recommendations = [
                                "1. **Focus on High Performers:** Prioritize candidates with 'Exceeds' ratings for leadership roles",
                                "2. **Engagement Potential:** Consider candidates with high engagement scores for team-building positions",
                                "3. **Department Balance:** Maintain diversity in selection across departments",
                                "4. **Development Opportunities:** Identify candidates with growth potential based on performance trends",
                                "5. **Risk Mitigation:** Monitor satisfaction scores to ensure long-term retention"
                            ]
                            for rec in recommendations:
                                st.write(rec)
                            
                            # Technical Notes
                            with st.expander("Technical Notes"):
                                st.write("""
                                - Analysis based on historical performance data
                                - Engagement and satisfaction scores normalized on a 1-5 scale
                                - Performance ratings weighted by recency
                                - Department distribution considered for balanced selection
                                - Key strengths identified through pattern analysis
                                """)
                    
                    else:
                        st.error(f"Analysis failed: {report_path}")
        else:
            st.info("👈 Select a scenario to continue")
    else:
        st.info("👈 Start by selecting a model")

def main():
    """Main UI flow"""
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Welcome", "Analysis"])
    
    # Display appropriate page
    if page == "Welcome":
        show_welcome()
    else:
        show_analysis()

if __name__ == "__main__":
    main() 