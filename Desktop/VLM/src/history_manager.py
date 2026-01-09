# Image History and Tracking System
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

class ImageHistoryManager:
    def __init__(self, history_file='data/image_history.json'):
        self.history_file = history_file
        self.history = self._load_history()
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load existing history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
        return []
    
    def _save_history(self):
        """Save history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def add_analysis(self, image_path: str, analysis_data: Dict[str, Any]) -> str:
        """Add new analysis to history"""
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        history_entry = {
            'id': analysis_id,
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'analysis': analysis_data,
            'follow_up_required': False,
            'treatment_effectiveness': None,
            'notes': ''
        }
        
        self.history.append(history_entry)
        
        # Keep only last 50 analyses to prevent file from growing too large
        if len(self.history) > 50:
            self.history = self.history[-50:]
        
        self._save_history()
        return analysis_id
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis history"""
        return sorted(self.history, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def get_analysis_by_id(self, analysis_id: str) -> Dict[str, Any]:
        """Get specific analysis by ID"""
        for entry in self.history:
            if entry['id'] == analysis_id:
                return entry
        return None
    
    def update_follow_up(self, analysis_id: str, follow_up_data: Dict[str, Any]):
        """Update follow-up information for an analysis"""
        for entry in self.history:
            if entry['id'] == analysis_id:
                entry['follow_up_required'] = True
                entry['follow_up_data'] = follow_up_data
                entry['follow_up_timestamp'] = datetime.now().isoformat()
                self._save_history()
                return True
        return False
    
    def get_disease_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected diseases"""
        disease_counts = {}
        severity_counts = {'none': 0, 'low': 0, 'moderate': 0, 'high': 0}
        
        for entry in self.history:
            if 'analysis' in entry and 'prediction' in entry['analysis']:
                disease = entry['analysis']['prediction']
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
                
                if 'severity' in entry['analysis']:
                    severity = entry['analysis']['severity']
                    if severity in severity_counts:
                        severity_counts[severity] += 1
        
        total_analyses = len(self.history)
        
        return {
            'total_analyses': total_analyses,
            'disease_counts': disease_counts,
            'severity_distribution': severity_counts,
            'most_common_disease': max(disease_counts.items(), key=lambda x: x[1])[0] if disease_counts else None,
            'average_confidence': sum(entry.get('analysis', {}).get('confidence', 0) for entry in self.history) / total_analyses if total_analyses > 0 else 0
        }
    
    def get_recent_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get disease detection trends over recent days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_entries = [entry for entry in self.history 
                        if datetime.fromisoformat(entry['timestamp']) > cutoff_date]
        
        daily_counts = {}
        for entry in recent_entries:
            date = entry['timestamp'][:10]  # YYYY-MM-DD
            daily_counts[date] = daily_counts.get(date, 0) + 1
        
        return {
            'period_days': days,
            'total_analyses': len(recent_entries),
            'daily_counts': daily_counts,
            'peak_day': max(daily_counts.items(), key=lambda x: x[1])[0] if daily_counts else None,
            'trend_direction': 'increasing' if len(daily_counts) > 1 and list(daily_counts.values())[-1] > list(daily_counts.values())[-2] else 'stable'
        }
