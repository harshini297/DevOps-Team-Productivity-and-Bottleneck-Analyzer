import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
from dataclasses import dataclass, asdict
import logging

# Import the analyzer classes from your original code
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data classes (from your original code)
@dataclass
class TeamMetrics:
    team_size: int
    sprint_duration: int
    total_commits: int
    deployment_freq: int
    lead_time: float
    failure_rate: float
    recovery_time: float
    code_review_time: float
    testing_time: float
    workflow_type: str
    tools_used: List[str]
    additional_notes: str = ""

@dataclass
class Bottleneck:
    type: str
    severity: str
    description: str
    impact_score: float
    suggested_actions: List[str]
    confidence: float = 0.0

@dataclass
class Recommendation:
    category: str
    action: str
    priority: str
    expected_impact: str
    implementation_effort: str

# All your analyzer classes (condensed for brevity - include all from original)
class DataCollector:
    def __init__(self):
        self.db_connection = self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect('devops_metrics.db', check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                team_size INTEGER,
                sprint_duration INTEGER,
                total_commits INTEGER,
                deployment_freq INTEGER,
                lead_time REAL,
                failure_rate REAL,
                recovery_time REAL,
                code_review_time REAL,
                testing_time REAL,
                workflow_type TEXT,
                tools_used TEXT,
                additional_notes TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metrics_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                productivity_score REAL,
                velocity_score REAL,
                quality_score REAL,
                efficiency_score REAL,
                bottlenecks TEXT,
                recommendations TEXT,
                FOREIGN KEY (metrics_id) REFERENCES team_metrics (id)
            )
        ''')
        conn.commit()
        return conn
    
    def store_metrics(self, metrics: TeamMetrics) -> int:
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO team_metrics (
                team_size, sprint_duration, total_commits, deployment_freq,
                lead_time, failure_rate, recovery_time, code_review_time,
                testing_time, workflow_type, tools_used, additional_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.team_size, metrics.sprint_duration, metrics.total_commits,
            metrics.deployment_freq, metrics.lead_time, metrics.failure_rate,
            metrics.recovery_time, metrics.code_review_time, metrics.testing_time,
            metrics.workflow_type, json.dumps(metrics.tools_used), metrics.additional_notes
        ))
        self.db_connection.commit()
        return cursor.lastrowid
    
    def get_historical_metrics(self, days: int = 30) -> pd.DataFrame:
        query = '''
            SELECT * FROM team_metrics 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)
        return pd.read_sql_query(query, self.db_connection)
    
    def get_analysis_results(self, days: int = 30) -> pd.DataFrame:
        query = '''
            SELECT ar.*, tm.timestamp as analysis_date
            FROM analysis_results ar
            JOIN team_metrics tm ON ar.metrics_id = tm.id
            WHERE tm.timestamp >= datetime('now', '-{} days')
            ORDER BY tm.timestamp DESC
        '''.format(days)
        return pd.read_sql_query(query, self.db_connection)

class MLWorkflowAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(random_state=42)
    
    def analyze_workflow(self, metrics: TeamMetrics, historical_data: pd.DataFrame) -> Dict:
        # Simplified version of your original method
        metrics_df = pd.DataFrame([[
            metrics.team_size, metrics.sprint_duration, metrics.total_commits,
            metrics.deployment_freq, metrics.lead_time, metrics.failure_rate,
            metrics.recovery_time, metrics.code_review_time, metrics.testing_time
        ]], columns=[
            'team_size', 'sprint_duration', 'total_commits', 'deployment_freq',
            'lead_time', 'failure_rate', 'recovery_time', 'code_review_time', 'testing_time'
        ])
        
        dora_metrics = self._calculate_dora_metrics(metrics)
        workflow_efficiency = self._calculate_workflow_efficiency(metrics)
        team_velocity = self._calculate_team_velocity(metrics)
        
        # Determine performance tier based on overall scores
        overall_score = (dora_metrics['overall_dora_score'] + 
                        workflow_efficiency['overall_efficiency'] + 
                        team_velocity['velocity_score']) / 3
        
        if overall_score >= 80:
            performance_tier = 'Elite'
        elif overall_score >= 60:
            performance_tier = 'High'
        else:
            performance_tier = 'Medium'
        
        return {
            'performance_tier': performance_tier,
            'dora_metrics': dora_metrics,
            'workflow_efficiency': workflow_efficiency,
            'team_velocity': team_velocity
        }
    
    def _calculate_dora_metrics(self, metrics: TeamMetrics) -> Dict:
        deploy_score = min(100, (metrics.deployment_freq / 5) * 100)
        lead_score = max(0, 100 - (metrics.lead_time / 168) * 100)
        failure_score = max(0, 100 - metrics.failure_rate * 2)
        recovery_score = max(0, 100 - (metrics.recovery_time / 24) * 100)
        return {
            'deployment_frequency_score': round(deploy_score, 2),
            'lead_time_score': round(lead_score, 2),
            'change_failure_rate_score': round(failure_score, 2),
            'recovery_time_score': round(recovery_score, 2),
            'overall_dora_score': round((deploy_score + lead_score + failure_score + recovery_score) / 4, 2)
        }
    
    def _calculate_workflow_efficiency(self, metrics: TeamMetrics) -> Dict:
        commits_per_dev = metrics.total_commits / metrics.team_size
        review_efficiency = max(0, 100 - (metrics.code_review_time / 24) * 100)
        testing_efficiency = max(0, 100 - (metrics.testing_time / 48) * 100)
        overall_efficiency = (commits_per_dev * 2 + review_efficiency + testing_efficiency) / 4
        return {
            'commits_per_developer': round(commits_per_dev, 2),
            'review_efficiency': round(review_efficiency, 2),
            'testing_efficiency': round(testing_efficiency, 2),
            'overall_efficiency': round(overall_efficiency, 2)
        }
    
    def _calculate_team_velocity(self, metrics: TeamMetrics) -> Dict:
        commits_per_day = metrics.total_commits / metrics.sprint_duration
        commits_per_dev_per_day = commits_per_day / metrics.team_size
        if commits_per_dev_per_day >= 3:
            velocity_score = 100
        elif commits_per_dev_per_day >= 2:
            velocity_score = 80
        elif commits_per_dev_per_day >= 1:
            velocity_score = 60
        else:
            velocity_score = 40
        return {
            'commits_per_day': round(commits_per_day, 2),
            'commits_per_dev_per_day': round(commits_per_dev_per_day, 2),
            'velocity_score': velocity_score
        }

class MLBottleneckDetector:
    def __init__(self):
        self.bottleneck_types = [
            'Lead Time', 'Deployment Quality', 'Deployment Frequency',
            'Recovery Time', 'Code Review Process', 'Testing Process', 'Team Velocity'
        ]
        self.thresholds = {
            'Lead Time': 72,
            'Deployment Quality': 10,
            'Deployment Frequency': 2,
            'Recovery Time': 8,
            'Code Review Process': 12,
            'Testing Process': 24,
            'Team Velocity': 10
        }
    
    def detect_bottlenecks(self, metrics: TeamMetrics) -> List[Bottleneck]:
        bottlenecks = []
        commits_per_dev = metrics.total_commits / metrics.team_size
        
        bottleneck_configs = {
            'Lead Time': {'value': metrics.lead_time, 'threshold': 72, 'is_higher': True},
            'Deployment Quality': {'value': metrics.failure_rate, 'threshold': 10, 'is_higher': True},
            'Deployment Frequency': {'value': metrics.deployment_freq, 'threshold': 2, 'is_higher': False},
            'Recovery Time': {'value': metrics.recovery_time, 'threshold': 8, 'is_higher': True},
            'Code Review Process': {'value': metrics.code_review_time, 'threshold': 12, 'is_higher': True},
            'Testing Process': {'value': metrics.testing_time, 'threshold': 24, 'is_higher': True},
            'Team Velocity': {'value': commits_per_dev, 'threshold': 10, 'is_higher': False}
        }
        
        for bottleneck_type, config in bottleneck_configs.items():
            value = config['value']
            threshold = config['threshold']
            is_higher = config['is_higher']
            
            if (is_higher and value > threshold) or (not is_higher and value < threshold):
                severity = 'High' if (is_higher and value > threshold * 1.5) or (not is_higher and value < threshold * 0.5) else 'Medium'
                impact_score = min(100, (abs(value - threshold) / threshold) * 100)
                
                actions = self._get_suggested_actions(bottleneck_type)
                description = self._get_bottleneck_description(bottleneck_type, value)
                
                bottlenecks.append(Bottleneck(
                    type=bottleneck_type,
                    severity=severity,
                    description=description,
                    impact_score=round(impact_score, 2),
                    suggested_actions=actions,
                    confidence=0.8
                ))
        
        return bottlenecks
    
    def _get_suggested_actions(self, bottleneck_type: str) -> List[str]:
        actions_map = {
            'Lead Time': ["Implement automated CI/CD pipelines", "Reduce batch sizes", "Optimize build processes"],
            'Deployment Quality': ["Enhance automated testing", "Implement blue-green deployments", "Add monitoring"],
            'Deployment Frequency': ["Adopt continuous deployment", "Use feature flags", "Reduce complexity"],
            'Recovery Time': ["Automate rollbacks", "Improve alerting", "Create playbooks"],
            'Code Review Process': ["Use automated tools", "Set SLAs", "Pair programming"],
            'Testing Process': ["Automate tests", "Parallel testing", "Optimize suite"],
            'Team Velocity': ["Remove blockers", "Improve tooling", "Retrospectives"]
        }
        return actions_map.get(bottleneck_type, ["General process improvement"])
    
    def _get_bottleneck_description(self, bottleneck_type: str, value: float) -> str:
        descriptions = {
            'Lead Time': f"Lead time of {value} hours exceeds optimal threshold",
            'Deployment Quality': f"Deployment failure rate of {value}% is too high",
            'Deployment Frequency': f"Low deployment frequency ({value}/week)",
            'Recovery Time': f"Mean recovery time of {value} hours is too high",
            'Code Review Process': f"Code review time of {value} hours creates delays",
            'Testing Process': f"Testing time of {value} hours slows delivery",
            'Team Velocity': f"Low commits per developer ({value:.1f})"
        }
        return descriptions.get(bottleneck_type, f"Issue detected with {bottleneck_type}")

class RecommendationEngine:
    def generate_recommendations(self, metrics: TeamMetrics, bottlenecks: List[Bottleneck]) -> List[Recommendation]:
        recommendations = []
        
        for bottleneck in bottlenecks:
            for action in bottleneck.suggested_actions:
                recommendations.append(Recommendation(
                    category=bottleneck.type,
                    action=action,
                    priority="High" if bottleneck.severity == "High" else "Medium",
                    expected_impact=self._estimate_impact(bottleneck.type),
                    implementation_effort=self._estimate_effort(action)
                ))
        
        return recommendations
    
    def _estimate_impact(self, bottleneck_type: str) -> str:
        impact_mapping = {
            'Lead Time': 'High - Faster delivery',
            'Deployment Quality': 'High - Reduced failures',
            'Deployment Frequency': 'Medium - Faster releases',
            'Recovery Time': 'High - Better reliability',
            'Code Review Process': 'Medium - Faster reviews',
            'Testing Process': 'Medium - Quality improvement',
            'Team Velocity': 'High - Productivity boost'
        }
        return impact_mapping.get(bottleneck_type, 'Medium - Process improvement')
    
    def _estimate_effort(self, action: str) -> str:
        if any(keyword in action.lower() for keyword in ['implement', 'create', 'automate']):
            return 'High'
        elif any(keyword in action.lower() for keyword in ['improve', 'enhance', 'optimize']):
            return 'Medium'
        else:
            return 'Low'

class ProductivityAnalyzer:
    def __init__(self):
        self.data_collector = DataCollector()
        self.workflow_analyzer = MLWorkflowAnalyzer()
        self.bottleneck_detector = MLBottleneckDetector()
        self.recommendation_engine = RecommendationEngine()
    
    def analyze_productivity(self, metrics_data: Dict) -> Dict:
        try:
            metrics = self._validate_metrics(metrics_data)
            metrics_id = self.data_collector.store_metrics(metrics)
            historical_data = self.data_collector.get_historical_metrics()
            
            workflow_analysis = self.workflow_analyzer.analyze_workflow(metrics, historical_data)
            bottlenecks = self.bottleneck_detector.detect_bottlenecks(metrics)
            recommendations = self.recommendation_engine.generate_recommendations(metrics, bottlenecks)
            scores = self._calculate_final_scores(workflow_analysis)
            
            self._store_analysis_results(metrics_id, scores, bottlenecks, recommendations)
            
            return {
                'scores': scores,
                'workflow_analysis': workflow_analysis,
                'bottlenecks': [asdict(b) for b in bottlenecks],
                'recommendations': [asdict(r) for r in recommendations],
                'metrics_id': metrics_id
            }
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _validate_metrics(self, metrics_data: Dict) -> TeamMetrics:
        required_fields = ['team_size', 'sprint_duration', 'total_commits', 'deployment_freq', 
                          'lead_time', 'failure_rate', 'recovery_time', 'code_review_time', 'testing_time']
        
        for field in required_fields:
            if field not in metrics_data:
                raise ValueError(f'Missing required field: {field}')
        
        return TeamMetrics(**metrics_data)
    
    def _calculate_final_scores(self, workflow_analysis: Dict) -> Dict:
        dora_score = workflow_analysis['dora_metrics']['overall_dora_score']
        efficiency_score = workflow_analysis['workflow_efficiency']['overall_efficiency']
        velocity_score = workflow_analysis['team_velocity']['velocity_score']
        
        return {
            'productivity': round((dora_score + efficiency_score) / 2, 1),
            'velocity': velocity_score,
            'quality': round((dora_score + efficiency_score) / 2, 1),
            'efficiency': round(efficiency_score, 1)
        }
    
    def _store_analysis_results(self, metrics_id: int, scores: Dict, bottlenecks: List[Bottleneck], recommendations: List[Recommendation]):
        cursor = self.data_collector.db_connection.cursor()
        cursor.execute('''
            INSERT INTO analysis_results (
                metrics_id, productivity_score, velocity_score, quality_score,
                efficiency_score, bottlenecks, recommendations
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics_id, scores['productivity'], scores['velocity'],
            scores['quality'], scores['efficiency'],
            json.dumps([asdict(b) for b in bottlenecks]),
            json.dumps([asdict(r) for r in recommendations])
        ))
        self.data_collector.db_connection.commit()

# Initialize the analyzer
@st.cache_resource
def get_analyzer():
    return ProductivityAnalyzer()

# Streamlit App Configuration
st.set_page_config(
    page_title="DevOps Productivity Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .bottleneck-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .bottleneck-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🚀 DevOps Productivity Analyzer</h1>', unsafe_allow_html=True)
    
    analyzer = get_analyzer()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Dashboard", 
        "📈 New Analysis", 
        "📋 Historical Data",
        "🔍 Detailed Analytics"
    ])
    
    if page == "📊 Dashboard":
        show_dashboard(analyzer)
    elif page == "📈 New Analysis":
        show_new_analysis(analyzer)
    elif page == "📋 Historical Data":
        show_historical_data(analyzer)
    elif page == "🔍 Detailed Analytics":
        show_detailed_analytics(analyzer)

def show_dashboard(analyzer):
    st.header("📊 Dashboard Overview")
    
    # Get recent analysis data
    recent_data = analyzer.data_collector.get_analysis_results(days=7)
    
    if recent_data.empty:
        st.info("No recent analysis data available. Run a new analysis to see dashboard metrics.")
        return
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    latest_scores = {
        'productivity': recent_data['productivity_score'].iloc[0],
        'velocity': recent_data['velocity_score'].iloc[0],
        'quality': recent_data['quality_score'].iloc[0],
        'efficiency': recent_data['efficiency_score'].iloc[0]
    }
    
    with col1:
        st.metric("Productivity Score", f"{latest_scores['productivity']:.1f}", 
                 delta=f"{latest_scores['productivity'] - 70:.1f}" if len(recent_data) > 1 else None)
    
    with col2:
        st.metric("Velocity Score", f"{latest_scores['velocity']:.1f}",
                 delta=f"{latest_scores['velocity'] - 60:.1f}" if len(recent_data) > 1 else None)
    
    with col3:
        st.metric("Quality Score", f"{latest_scores['quality']:.1f}",
                 delta=f"{latest_scores['quality'] - 75:.1f}" if len(recent_data) > 1 else None)
    
    with col4:
        st.metric("Efficiency Score", f"{latest_scores['efficiency']:.1f}",
                 delta=f"{latest_scores['efficiency'] - 65:.1f}" if len(recent_data) > 1 else None)
    
    # Trend charts
    st.subheader("📈 Trends Over Time")
    
    if len(recent_data) > 1:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Productivity', 'Velocity', 'Quality', 'Efficiency'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Add trend lines
        fig.add_trace(
            go.Scatter(x=recent_data['analysis_date'], y=recent_data['productivity_score'],
                      mode='lines+markers', name='Productivity'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=recent_data['analysis_date'], y=recent_data['velocity_score'],
                      mode='lines+markers', name='Velocity'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=recent_data['analysis_date'], y=recent_data['quality_score'],
                      mode='lines+markers', name='Quality'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=recent_data['analysis_date'], y=recent_data['efficiency_score'],
                      mode='lines+markers', name='Efficiency'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent bottlenecks
    st.subheader("🚨 Recent Bottlenecks")
    if not recent_data.empty and recent_data['bottlenecks'].iloc[0]:
        bottlenecks = json.loads(recent_data['bottlenecks'].iloc[0])
        for bottleneck in bottlenecks[:3]:  # Show top 3 bottlenecks
            severity_class = "bottleneck-high" if bottleneck['severity'] == 'High' else "bottleneck-medium"
            st.markdown(f"""
            <div class="{severity_class}">
                <strong>{bottleneck['type']}</strong> ({bottleneck['severity']} Priority)<br>
                {bottleneck['description']}<br>
                <small>Impact Score: {bottleneck['impact_score']:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent bottlenecks detected.")

def show_new_analysis(analyzer):
    st.header("📈 New Team Analysis")
    
    with st.form("metrics_form"):
        st.subheader("Team Metrics Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            team_size = st.number_input("Team Size", min_value=1, max_value=50, value=5)
            sprint_duration = st.number_input("Sprint Duration (days)", min_value=1, max_value=30, value=14)
            total_commits = st.number_input("Total Commits", min_value=0, value=100)
            deployment_freq = st.number_input("Deployment Frequency (per week)", min_value=0, value=3)
            workflow_type = st.selectbox("Workflow Type", ["Agile", "Waterfall", "DevOps", "Hybrid"])
        
        with col2:
            lead_time = st.number_input("Lead Time (hours)", min_value=0.0, value=48.0)
            failure_rate = st.slider("Failure Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
            recovery_time = st.number_input("Recovery Time (hours)", min_value=0.0, value=4.0)
            code_review_time = st.number_input("Code Review Time (hours)", min_value=0.0, value=8.0)
            testing_time = st.number_input("Testing Time (hours)", min_value=0.0, value=16.0)
        
        # Tools used
        st.subheader("Tools & Technologies")
        tools_options = ["Jenkins", "GitLab CI", "GitHub Actions", "Docker", "Kubernetes", 
                        "Terraform", "Ansible", "Prometheus", "Grafana", "SonarQube"]
        tools_used = st.multiselect("Select tools used", tools_options, default=["Jenkins", "Docker"])
        
        # Additional notes
        additional_notes = st.text_area("Additional Notes", placeholder="Any specific context or information about your team...")
        
        submitted = st.form_submit_button("🔍 Analyze Team Productivity")
        
        if submitted:
            with st.spinner("Analyzing team productivity..."):
                metrics_data = {
                    'team_size': team_size,
                    'sprint_duration': sprint_duration,
                    'total_commits': total_commits,
                    'deployment_freq': deployment_freq,
                    'lead_time': lead_time,
                    'failure_rate': failure_rate,
                    'recovery_time': recovery_time,
                    'code_review_time': code_review_time,
                    'testing_time': testing_time,
                    'workflow_type': workflow_type,
                    'tools_used': tools_used,
                    'additional_notes': additional_notes
                }
                
                results = analyzer.analyze_productivity(metrics_data)
                
                if 'error' in results:
                    st.error(f"Analysis failed: {results['error']}")
                else:
                    show_analysis_results(results)

def show_analysis_results(results):
    st.success("✅ Analysis completed successfully!")
    
    # Display scores
    st.subheader("📊 Performance Scores")
    col1, col2, col3, col4 = st.columns(4)
    
    scores = results['scores']
    
    with col1:
        st.metric("Productivity", f"{scores['productivity']:.1f}/100")
    with col2:
        st.metric("Velocity", f"{scores['velocity']:.1f}/100")
    with col3:
        st.metric("Quality", f"{scores['quality']:.1f}/100")
    with col4:
        st.metric("Efficiency", f"{scores['efficiency']:.1f}/100")
    
    # Performance tier
    performance_tier = results['workflow_analysis']['performance_tier']
    tier_color = {"Elite": "🟢", "High": "🟡", "Medium": "🟠", "Low": "🔴"}
    st.info(f"**Performance Tier:** {tier_color.get(performance_tier, '⚪')} {performance_tier}")
    
    # DORA Metrics
    st.subheader("🎯 DORA Metrics Breakdown")
    dora_metrics = results['workflow_analysis']['dora_metrics']
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Deployment Frequency Score:**", f"{dora_metrics['deployment_frequency_score']:.1f}/100")
        st.write("**Lead Time Score:**", f"{dora_metrics['lead_time_score']:.1f}/100")
    with col2:
        st.write("**Change Failure Rate Score:**", f"{dora_metrics['change_failure_rate_score']:.1f}/100")
        st.write("**Recovery Time Score:**", f"{dora_metrics['recovery_time_score']:.1f}/100")
    
    # Radar chart for DORA metrics
    categories = ['Deployment Frequency', 'Lead Time', 'Change Failure Rate', 'Recovery Time']
    values = [
        dora_metrics['deployment_frequency_score'],
        dora_metrics['lead_time_score'],
        dora_metrics['change_failure_rate_score'],
        dora_metrics['recovery_time_score']
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='DORA Metrics'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="DORA Metrics Radar Chart"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Bottlenecks
    st.subheader("🚨 Identified Bottlenecks")
    bottlenecks = results['bottlenecks']
    
    if bottlenecks:
        for bottleneck in bottlenecks:
            severity_class = "bottleneck-high" if bottleneck['severity'] == 'High' else "bottleneck-medium"
            st.markdown(f"""
            <div class="{severity_class}">
                <strong>{bottleneck['type']}</strong> ({bottleneck['severity']} Priority)<br>
                {bottleneck['description']}<br>
                <small>Impact Score: {bottleneck['impact_score']:.1f}% | Confidence: {bottleneck['confidence']:.1f}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Show suggested actions
            st.write("**Suggested Actions:**")
            for action in bottleneck['suggested_actions']:
                st.write(f"• {action}")
    else:
        st.success("🎉 No significant bottlenecks detected!")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    recommendations = results['recommendations']
    
    if recommendations:
        for rec in recommendations:
            st.markdown(f"""
            <div class="recommendation">
                <strong>{rec['category']}</strong> - {rec['priority']} Priority<br>
                <strong>Action:</strong> {rec['action']}<br>
                <strong>Expected Impact:</strong> {rec['expected_impact']}<br>
                <small>Implementation Effort: {rec['implementation_effort']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No specific recommendations at this time.")
    
    # Detailed metrics
    with st.expander("🔍 Detailed Metrics"):
        workflow_analysis = results['workflow_analysis']
        
        st.write("**Workflow Efficiency:**")
        efficiency = workflow_analysis['workflow_efficiency']
        st.json(efficiency)
        
        st.write("**Team Velocity:**")
        velocity = workflow_analysis['team_velocity']
        st.json(velocity)

def show_historical_data(analyzer):
    st.header("📋 Historical Analysis Data")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.selectbox("Time Range", [7, 14, 30, 60, 90], index=2)
    with col2:
        data_type = st.selectbox("Data Type", ["Metrics", "Analysis Results", "Both"])
    
    if data_type in ["Metrics", "Both"]:
        st.subheader("📊 Historical Metrics")
        metrics_data = analyzer.data_collector.get_historical_metrics(days=days_back)
        
        if not metrics_data.empty:
            # Clean up the data for display
            display_data = metrics_data.copy()
            display_data['tools_used'] = display_data['tools_used'].apply(
                lambda x: ', '.join(json.loads(x)) if x else ''
            )
            
            st.dataframe(display_data, use_container_width=True)
            
            # Download button
            csv = display_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Metrics CSV",
                data=csv,
                file_name=f"team_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"No metrics data found for the last {days_back} days.")
    
    if data_type in ["Analysis Results", "Both"]:
        st.subheader("🔍 Historical Analysis Results")
        analysis_data = analyzer.data_collector.get_analysis_results(days=days_back)
        
        if not analysis_data.empty:
            # Show summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_productivity = analysis_data['productivity_score'].mean()
                st.metric("Avg Productivity", f"{avg_productivity:.1f}")
            
            with col2:
                avg_velocity = analysis_data['velocity_score'].mean()
                st.metric("Avg Velocity", f"{avg_velocity:.1f}")
            
            with col3:
                avg_quality = analysis_data['quality_score'].mean()
                st.metric("Avg Quality", f"{avg_quality:.1f}")
            
            with col4:
                avg_efficiency = analysis_data['efficiency_score'].mean()
                st.metric("Avg Efficiency", f"{avg_efficiency:.1f}")
            
            # Trend visualization
            st.subheader("📈 Score Trends")
            fig = px.line(
                analysis_data, 
                x='analysis_date', 
                y=['productivity_score', 'velocity_score', 'quality_score', 'efficiency_score'],
                title="Performance Scores Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            display_analysis = analysis_data[['analysis_date', 'productivity_score', 'velocity_score', 
                                           'quality_score', 'efficiency_score']].copy()
            st.dataframe(display_analysis, use_container_width=True)
            
        else:
            st.info(f"No analysis results found for the last {days_back} days.")

def show_detailed_analytics(analyzer):
    st.header("🔍 Detailed Analytics")
    
    # Get data for analytics
    days_back = st.slider("Analysis Period (days)", min_value=7, max_value=90, value=30)
    
    historical_data = analyzer.data_collector.get_historical_metrics(days=days_back)
    analysis_data = analyzer.data_collector.get_analysis_results(days=days_back)
    
    if historical_data.empty or analysis_data.empty:
        st.warning("Insufficient data for detailed analytics. Please run more analyses.")
        return
    
    # Correlation analysis
    st.subheader("📊 Correlation Analysis")
    
    # Prepare data for correlation
    numeric_cols = ['team_size', 'sprint_duration', 'total_commits', 'deployment_freq', 
                   'lead_time', 'failure_rate', 'recovery_time', 'code_review_time', 'testing_time']
    
    if len(historical_data) > 1:
        correlation_data = historical_data[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_data,
            text_auto=True,
            aspect="auto",
            title="Metrics Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance distribution
    st.subheader("📈 Performance Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            analysis_data, 
            x='productivity_score', 
            nbins=20,
            title="Productivity Score Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            analysis_data, 
            x='velocity_score', 
            nbins=20,
            title="Velocity Score Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Team size vs performance
    st.subheader("👥 Team Size Impact")
    
    if len(historical_data) > 1:
        # Merge historical and analysis data
        merged_data = pd.merge(historical_data, analysis_data, left_on='id', right_on='metrics_id', how='inner')
        
        fig = px.scatter(
            merged_data,
            x='team_size',
            y='productivity_score',
            size='total_commits',
            color='velocity_score',
            title="Team Size vs Productivity (bubble size = commits, color = velocity)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottleneck frequency analysis
    st.subheader("🚨 Bottleneck Analysis")
    
    if not analysis_data.empty:
        bottleneck_counts = {}
        
        for _, row in analysis_data.iterrows():
            if row['bottlenecks']:
                bottlenecks = json.loads(row['bottlenecks'])
                for bottleneck in bottlenecks:
                    bottleneck_type = bottleneck['type']
                    if bottleneck_type not in bottleneck_counts:
                        bottleneck_counts[bottleneck_type] = 0
                    bottleneck_counts[bottleneck_type] += 1
        
        if bottleneck_counts:
            bottleneck_df = pd.DataFrame(list(bottleneck_counts.items()), 
                                       columns=['Bottleneck Type', 'Frequency'])
            
            fig = px.bar(
                bottleneck_df,
                x='Bottleneck Type',
                y='Frequency',
                title="Most Common Bottlenecks"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No bottlenecks found in the analysis period.")
    
    # Improvement recommendations
    st.subheader("💡 Improvement Insights")
    
    if not analysis_data.empty:
        latest_scores = analysis_data.iloc[0]
        
        insights = []
        
        if latest_scores['productivity_score'] < 70:
            insights.append("🔴 **Low Productivity Alert**: Focus on reducing lead time and improving deployment frequency")
        
        if latest_scores['velocity_score'] < 60:
            insights.append("🟡 **Velocity Improvement**: Consider increasing team size or removing blockers")
        
        if latest_scores['quality_score'] < 75:
            insights.append("🟠 **Quality Focus**: Invest in better testing and code review processes")
        
        if latest_scores['efficiency_score'] < 65:
            insights.append("⚪ **Efficiency Optimization**: Automate repetitive tasks and streamline workflows")
        
        if not insights:
            insights.append("🟢 **Great Performance**: Team is performing well across all metrics!")
        
        for insight in insights:
            st.markdown(insight)

# Advanced features
def show_prediction_model(analyzer):
    """Future feature: Predictive analytics"""
    st.subheader("🔮 Predictive Analytics (Coming Soon)")
    st.info("This feature will predict future performance based on historical trends.")

def show_team_comparison(analyzer):
    """Future feature: Team comparison"""
    st.subheader("⚖️ Team Comparison (Coming Soon)")
    st.info("This feature will allow comparison between multiple teams.")

# Export functionality
def export_analysis_report(results, team_name="Team"):
    """Generate a comprehensive report"""
    report = f"""
# DevOps Productivity Analysis Report
## Team: {team_name}
## Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Performance Scores
- Productivity: {results['scores']['productivity']:.1f}/100
- Velocity: {results['scores']['velocity']:.1f}/100
- Quality: {results['scores']['quality']:.1f}/100
- Efficiency: {results['scores']['efficiency']:.1f}/100

### Performance Tier
{results['workflow_analysis']['performance_tier']}

### DORA Metrics
- Deployment Frequency Score: {results['workflow_analysis']['dora_metrics']['deployment_frequency_score']:.1f}/100
- Lead Time Score: {results['workflow_analysis']['dora_metrics']['lead_time_score']:.1f}/100
- Change Failure Rate Score: {results['workflow_analysis']['dora_metrics']['change_failure_rate_score']:.1f}/100
- Recovery Time Score: {results['workflow_analysis']['dora_metrics']['recovery_time_score']:.1f}/100

### Identified Bottlenecks
"""
    
    for bottleneck in results['bottlenecks']:
        report += f"""
#### {bottleneck['type']} ({bottleneck['severity']} Priority)
- Description: {bottleneck['description']}
- Impact Score: {bottleneck['impact_score']:.1f}%
- Suggested Actions:
"""
        for action in bottleneck['suggested_actions']:
            report += f"  - {action}\n"
    
    report += "\n### Recommendations\n"
    for rec in results['recommendations']:
        report += f"""
#### {rec['category']} - {rec['priority']} Priority
- Action: {rec['action']}
- Expected Impact: {rec['expected_impact']}
- Implementation Effort: {rec['implementation_effort']}
"""
    
    return report

# Main execution
if __name__ == "__main__":
    main()
