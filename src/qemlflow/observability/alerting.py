"""
Alerting System Module

This module provides comprehensive alerting capabilities including notification
channels, alert routing, escalation policies, and integration with external
systems for enterprise-grade alert management.
"""

import json
import logging
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from .monitoring import Alert, AlertRule

try:
    import requests
except ImportError:
    requests = None


class NotificationChannel(ABC):
    """Base class for notification channels."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    async def send_notification(self, alert: Alert, message: str) -> bool:
        """Send notification for alert."""
        pass
    
    def format_alert_message(self, alert: Alert) -> str:
        """Format alert message."""
        return (
            f"Alert: {alert.name}\n"
            f"Severity: {alert.severity.upper()}\n"
            f"Current Value: {alert.current_value}\n"
            f"Threshold: {alert.threshold}\n"
            f"Time: {alert.triggered_at}\n"
            f"Message: {alert.message}\n"
        )


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, name: str = "email", smtp_server: str = "localhost",
                 smtp_port: int = 587, username: str = "", password: str = "",
                 from_email: str = "", use_tls: bool = True, enabled: bool = True):
        super().__init__(name, enabled)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.use_tls = use_tls
    
    async def send_notification(self, alert: Alert, message: str = "") -> bool:
        """Send email notification."""
        if not self.enabled:
            return False
        
        try:
            # Get recipients from alert rule or configuration
            recipients = getattr(alert, 'email_recipients', [])
            if not recipients:
                self.logger.warning(f"No email recipients configured for alert {alert.alert_id}")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.upper()}] QeMLflow Alert: {alert.name}"
            
            body = message or self.format_alert_message(alert)
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)
            
            self.logger.info(f"Email notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(self, name: str = "slack", webhook_url: str = "",
                 channel: str = "#alerts", enabled: bool = True):
        super().__init__(name, enabled)
        self.webhook_url = webhook_url
        self.channel = channel
    
    async def send_notification(self, alert: Alert, message: str = "") -> bool:
        """Send Slack notification."""
        if not self.enabled or not self.webhook_url or not requests:
            return False
        
        try:
            # Format Slack message
            color = {
                "info": "#36a64f",      # Green
                "warning": "#ffcc00",   # Yellow
                "critical": "#ff0000"   # Red
            }.get(alert.severity, "#cccccc")
            
            payload = {
                "channel": self.channel,
                "username": "QeMLflow Monitor",
                "attachments": [{
                    "color": color,
                    "title": f"Alert: {alert.name}",
                    "fields": [
                        {"title": "Severity", "value": alert.severity.upper(), "short": True},
                        {"title": "Current Value", "value": str(alert.current_value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold), "short": True},
                        {"title": "Status", "value": alert.status, "short": True},
                        {"title": "Time", "value": alert.triggered_at, "short": False},
                        {"title": "Message", "value": message or alert.message, "short": False}
                    ]
                }]
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Slack notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel."""
    
    def __init__(self, name: str = "webhook", url: str = "", 
                 headers: Optional[Dict[str, str]] = None, enabled: bool = True):
        super().__init__(name, enabled)
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
    
    async def send_notification(self, alert: Alert, message: str = "") -> bool:
        """Send webhook notification."""
        if not self.enabled or not self.url or not requests:
            return False
        
        try:
            payload = {
                "alert": alert.to_dict(),
                "message": message or alert.message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            response = requests.post(
                self.url, 
                json=payload, 
                headers=self.headers, 
                timeout=10
            )
            response.raise_for_status()
            
            self.logger.info(f"Webhook notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            return False


@dataclass
class EscalationPolicy:
    """Alert escalation policy."""
    
    policy_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    
    # Escalation levels (in minutes)
    level_1_delay: int = 5
    level_2_delay: int = 15
    level_3_delay: int = 30
    
    # Notification channels for each level
    level_1_channels: List[str] = field(default_factory=list)
    level_2_channels: List[str] = field(default_factory=list)
    level_3_channels: List[str] = field(default_factory=list)
    
    # Conditions
    apply_to_severities: List[str] = field(default_factory=lambda: ["critical"])
    apply_to_services: List[str] = field(default_factory=list)
    
    enabled: bool = True


@dataclass 
class AlertHistory:
    """Alert history tracking."""
    
    alert_id: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_event(self, event_type: str, details: Optional[Dict[str, Any]] = None):
        """Add event to history."""
        self.events.append({
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {}
        })


class AlertManager:
    """Central alert management system."""
    
    def __init__(self, storage_dir: str = "alerts"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        # State
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: Dict[str, AlertHistory] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        
        # Notification channels
        self.notification_channels: Dict[str, NotificationChannel] = {}
        
        # Suppression
        self.suppressed_rules: Set[str] = set()
        self.maintenance_mode: bool = False
        
        self.logger = logging.getLogger(__name__)
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add notification channel."""
        self.notification_channels[channel.name] = channel
        self.logger.info(f"Added notification channel: {channel.name}")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def add_escalation_policy(self, policy: EscalationPolicy):
        """Add escalation policy."""
        self.escalation_policies[policy.policy_id] = policy
        self.logger.info(f"Added escalation policy: {policy.name}")
    
    def evaluate_rules(self, metrics: Dict[str, float]):
        """Evaluate alert rules against current metrics."""
        if self.maintenance_mode:
            return
        
        for rule in self.alert_rules.values():
            if not rule.enabled or rule.rule_id in self.suppressed_rules:
                continue
            
            metric_value = metrics.get(rule.metric_name)
            if metric_value is None:
                continue
            
            should_trigger = rule.evaluate(metric_value)
            existing_alert = self._find_active_alert_for_rule(rule.rule_id)
            
            if should_trigger and not existing_alert:
                self._trigger_alert(rule, metric_value)
            elif not should_trigger and existing_alert:
                self._resolve_alert(existing_alert)
    
    def _find_active_alert_for_rule(self, rule_id: str) -> Optional[Alert]:
        """Find active alert for rule."""
        for alert in self.active_alerts.values():
            if alert.rule_id == rule_id and alert.status == "active":
                return alert
        return None
    
    def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger new alert."""
        alert = Alert(
            rule_id=rule.rule_id,
            name=rule.name,
            message=f"{rule.metric_name} {rule.condition} {rule.threshold}",
            severity=rule.severity,
            current_value=current_value,
            threshold=rule.threshold
        )
        
        self.active_alerts[alert.alert_id] = alert
        
        # Add to history
        self.alert_history[alert.alert_id] = AlertHistory(alert.alert_id)
        self.alert_history[alert.alert_id].add_event("triggered", {
            "rule_id": rule.rule_id,
            "current_value": current_value,
            "threshold": rule.threshold
        })
        
        # Send notifications
        self._send_alert_notifications(alert, rule)
        
        self.logger.warning(f"Alert triggered: {alert.name} (ID: {alert.alert_id})")
    
    def _resolve_alert(self, alert: Alert):
        """Resolve active alert."""
        alert.resolve()
        
        # Add to history
        if alert.alert_id in self.alert_history:
            self.alert_history[alert.alert_id].add_event("resolved")
        
        self.logger.info(f"Alert resolved: {alert.name} (ID: {alert.alert_id})")
    
    def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for alert."""
        # Email notifications
        if rule.email_recipients:
            email_channel = self.notification_channels.get("email")
            if email_channel:
                # Store recipients in alert for channel access
                alert.email_recipients = rule.email_recipients
                # Note: In real implementation, this would be async
                # asyncio.create_task(email_channel.send_notification(alert))
        
        # Slack notifications
        if rule.slack_channel:
            slack_channel = self.notification_channels.get("slack")
            if slack_channel:
                # Note: In real implementation, this would be async
                # asyncio.create_task(slack_channel.send_notification(alert))
                pass
        
        # Webhook notifications
        if rule.webhook_url:
            webhook_channel = self.notification_channels.get("webhook")
            if webhook_channel:
                # Note: In real implementation, this would be async
                # asyncio.create_task(webhook_channel.send_notification(alert))
                pass
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [alert for alert in self.active_alerts.values() if alert.status == "active"]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        active_alerts = self.get_active_alerts()
        
        severity_counts = {"info": 0, "warning": 0, "critical": 0}
        for alert in active_alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        return {
            "total_active": len(active_alerts),
            "severity_breakdown": severity_counts,
            "maintenance_mode": self.maintenance_mode,
            "suppressed_rules": len(self.suppressed_rules),
            "total_rules": len(self.alert_rules),
            "notification_channels": len(self.notification_channels)
        }
    
    def suppress_rule(self, rule_id: str, duration_minutes: int = 60):
        """Temporarily suppress alert rule."""
        self.suppressed_rules.add(rule_id)
        self.logger.info(f"Suppressed rule {rule_id} for {duration_minutes} minutes")
        
        # TODO: Implement timer to automatically unsuppress
    
    def unsuppress_rule(self, rule_id: str):
        """Remove rule suppression."""
        self.suppressed_rules.discard(rule_id)
        self.logger.info(f"Unsuppressed rule {rule_id}")
    
    def enter_maintenance_mode(self, duration_minutes: int = 60):
        """Enter maintenance mode (suppress all alerts)."""
        self.maintenance_mode = True
        self.logger.info(f"Entered maintenance mode for {duration_minutes} minutes")
        
        # TODO: Implement timer to automatically exit maintenance mode
    
    def exit_maintenance_mode(self):
        """Exit maintenance mode."""
        self.maintenance_mode = False
        self.logger.info("Exited maintenance mode")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledge()
            
            if alert_id in self.alert_history:
                self.alert_history[alert_id].add_event("acknowledged")
            
            self.logger.info(f"Acknowledged alert: {alert_id}")
            return True
        
        return False
    
    def save_state(self):
        """Save alert manager state to disk."""
        state = {
            "active_alerts": {k: v.to_dict() for k, v in self.active_alerts.items()},
            "alert_rules": {k: asdict(v) for k, v in self.alert_rules.items()},
            "escalation_policies": {k: asdict(v) for k, v in self.escalation_policies.items()},
            "suppressed_rules": list(self.suppressed_rules),
            "maintenance_mode": self.maintenance_mode
        }
        
        state_file = self.storage_dir / "alert_manager_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        """Load alert manager state from disk."""
        state_file = self.storage_dir / "alert_manager_state.json"
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Restore active alerts
            for alert_id, alert_data in state.get("active_alerts", {}).items():
                alert = Alert(**alert_data)
                self.active_alerts[alert_id] = alert
            
            # Restore alert rules
            for rule_id, rule_data in state.get("alert_rules", {}).items():
                rule = AlertRule(**rule_data)
                self.alert_rules[rule_id] = rule
            
            # Restore escalation policies
            for policy_id, policy_data in state.get("escalation_policies", {}).items():
                policy = EscalationPolicy(**policy_data)
                self.escalation_policies[policy_id] = policy
            
            # Restore other state
            self.suppressed_rules = set(state.get("suppressed_rules", []))
            self.maintenance_mode = state.get("maintenance_mode", False)
            
            self.logger.info("Alert manager state loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load alert manager state: {e}")


# Global alert manager instance
_alert_manager = None

def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
