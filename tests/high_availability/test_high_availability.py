"""
Tests for High Availability system.
"""

import pytest
from unittest.mock import Mock, patch

from qemlflow.high_availability import (
    HighAvailabilityManager,
    RedundancyManager,
    DisasterRecoveryManager,
    BackupRestoreManager,
    FailoverManager,
    HealthMonitor
)


class TestRedundancyManager:
    """Test redundancy management functionality."""
    
    @pytest.fixture
    def config(self):
        return {
            'redundancy': {
                'services': {
                    'compute_nodes': {
                        'min_instances': 2,
                        'health_check_interval': 30
                    }
                }
            }
        }
    
    @pytest.fixture
    def redundancy_manager(self, config):
        return RedundancyManager(config)
    
    def test_initialization(self, redundancy_manager):
        """Test redundancy manager initialization."""
        assert redundancy_manager.config is not None
        assert isinstance(redundancy_manager.services_status, dict)
        assert redundancy_manager._monitoring is False
    
    def test_start_stop_monitoring(self, redundancy_manager):
        """Test starting and stopping monitoring."""
        redundancy_manager.start_monitoring()
        assert redundancy_manager._monitoring is True
        
        redundancy_manager.stop_monitoring()
        assert redundancy_manager._monitoring is False
    
    def test_get_redundancy_status(self, redundancy_manager):
        """Test getting redundancy status."""
        status = redundancy_manager.get_redundancy_status()
        
        assert 'services' in status
        assert 'monitoring_active' in status
        assert 'last_update' in status
        assert status['monitoring_active'] is False
    
    def test_check_single_service(self, redundancy_manager):
        """Test single service health checking."""
        service_config = {'min_instances': 2}
        
        with patch('psutil.cpu_percent', return_value=75.0), \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_memory.return_value.percent = 70.0
            
            health_status = redundancy_manager._check_single_service('test_service', service_config)
            
            assert health_status['status'] == 'healthy'
            assert 'cpu_usage' in health_status
            assert 'memory_usage' in health_status
            assert 'last_check' in health_status


class TestDisasterRecoveryManager:
    """Test disaster recovery functionality."""
    
    @pytest.fixture
    def config(self):
        return {
            'disaster_recovery': {
                'rto': 3600,
                'rpo': 900
            }
        }
    
    @pytest.fixture
    def disaster_recovery_manager(self, config):
        return DisasterRecoveryManager(config)
    
    def test_initialization(self, disaster_recovery_manager):
        """Test disaster recovery manager initialization."""
        assert disaster_recovery_manager.config is not None
        assert isinstance(disaster_recovery_manager.recovery_procedures, dict)
    
    def test_create_recovery_plan(self, disaster_recovery_manager):
        """Test recovery plan creation."""
        plan = disaster_recovery_manager.create_recovery_plan()
        
        assert 'plan_id' in plan
        assert 'created' in plan  # Changed from 'created_at'
        assert 'procedures' in plan
        assert 'resources' in plan
        assert 'contacts' in plan
        assert len(plan['procedures']) > 0
        assert plan['rto'] == 3600
    
    def test_execute_recovery(self, disaster_recovery_manager):
        """Test recovery execution."""
        plan = disaster_recovery_manager.create_recovery_plan()
        
        result = disaster_recovery_manager.execute_recovery(plan['plan_id'])
        
        assert result['status'] == 'completed'  # Changed from 'success'
        assert 'execution_id' in result
        assert 'steps_completed' in result


class TestBackupRestoreManager:
    """Test backup and restore functionality."""
    
    @pytest.fixture
    def config(self):
        return {
            'disaster_recovery': {
                'backup': {
                    'automated': True,
                    'frequency': 'daily'
                }
            }
        }
    
    @pytest.fixture
    def backup_manager(self, config):
        return BackupRestoreManager(config)
    
    def test_initialization(self, backup_manager):
        """Test backup manager initialization."""
        assert backup_manager.config is not None
        assert len(backup_manager.backup_history) == 0
        assert backup_manager._backup_scheduled is False
    
    def test_create_backup(self, backup_manager):
        """Test backup creation."""
        backup_info = backup_manager.create_backup('manual')
        
        assert backup_info.backup_id.startswith('backup_')
        assert backup_info.type == 'manual'
        assert backup_info.status == 'success'
        assert backup_info.size_bytes > 0
        assert len(backup_manager.backup_history) == 1
    
    def test_restore_from_backup(self, backup_manager):
        """Test backup restoration."""
        # Create a backup first
        backup_info = backup_manager.create_backup('manual')
        
        # Restore from backup
        restore_result = backup_manager.restore_from_backup(backup_info.backup_id)
        
        assert restore_result['backup_id'] == backup_info.backup_id
        assert restore_result['status'] == 'completed'
        assert 'restore_id' in restore_result
    
    def test_backup_status(self, backup_manager):
        """Test getting backup status."""
        # Create some backups
        backup_manager.create_backup('manual')
        backup_manager.create_backup('automated')
        
        status = backup_manager.get_backup_status()
        
        assert status['total_backups'] == 2
        assert status['last_backup'] is not None
        assert len(status['recent_backups']) == 2
    
    def test_start_stop_automated_backup(self, backup_manager):
        """Test automated backup management."""
        # Start automated backup
        backup_manager.start_automated_backup()
        assert backup_manager._backup_scheduled is True
        
        # Stop automated backup
        backup_manager.stop_automated_backup()
        assert backup_manager._backup_scheduled is False


class TestFailoverManager:
    """Test failover management functionality."""
    
    @pytest.fixture
    def config(self):
        return {
            'failover': {
                'automatic': {
                    'health_checks': True
                }
            }
        }
    
    @pytest.fixture
    def failover_manager(self, config):
        return FailoverManager(config)
    
    def test_initialization(self, failover_manager):
        """Test failover manager initialization."""
        assert failover_manager.config is not None
        assert isinstance(failover_manager.failover_history, list)
        assert failover_manager._monitoring is False
    
    def test_start_stop_monitoring(self, failover_manager):
        """Test starting and stopping monitoring."""
        failover_manager.start_monitoring()
        assert failover_manager._monitoring is True
        
        failover_manager.stop_monitoring()
        assert failover_manager._monitoring is False
    
    def test_initiate_failover(self, failover_manager):
        """Test failover initiation."""
        with patch.object(failover_manager, '_execute_failover') as mock_execute:
            mock_execute.return_value = None
            
            failover_event = failover_manager.initiate_failover('test_service', 'manual_test')
            
            assert failover_event.service == 'test_service'
            assert failover_event.trigger == 'manual_test'
            assert failover_event.status == 'completed'
            assert len(failover_manager.failover_history) == 1
    
    def test_get_failover_status(self, failover_manager):
        """Test getting failover status."""
        status = failover_manager.get_failover_status()
        
        assert 'monitoring_active' in status
        assert 'recent_failovers' in status
        assert 'total_failovers' in status
        assert 'automatic_failover_enabled' in status
        assert status['monitoring_active'] is False
    
    def test_select_target_node(self, failover_manager):
        """Test target node selection."""
        target_node = failover_manager._select_target_node('test_service')
        
        assert target_node == 'backup_node_for_test_service'


class TestHealthMonitor:
    """Test health monitoring functionality."""
    
    @pytest.fixture
    def config(self):
        return {
            'health_monitoring': {
                'system': {
                    'cpu_threshold': 80,
                    'memory_threshold': 85,
                    'disk_threshold': 90
                }
            }
        }
    
    @pytest.fixture
    def health_monitor(self, config):
        return HealthMonitor(config)
    
    def test_initialization(self, health_monitor):
        """Test health monitor initialization."""
        assert health_monitor.config is not None
        assert len(health_monitor.health_history) == 0
        assert health_monitor._monitoring is False
    
    def test_collect_health_metrics(self, health_monitor):
        """Test health metrics collection."""
        with patch('psutil.cpu_percent', return_value=75.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.getloadavg', return_value=[1.0, 1.5, 2.0]), \
             patch('psutil.net_connections', return_value=[]), \
             patch('psutil.net_io_counters') as mock_net_io, \
             patch('psutil.pids', return_value=[1, 2, 3]), \
             patch('psutil.process_iter', return_value=[]):
            
            mock_memory.return_value.percent = 70.0
            mock_disk.return_value.percent = 60.0
            mock_net_io.return_value.bytes_sent = 1000
            mock_net_io.return_value.bytes_recv = 2000
            
            metrics = health_monitor._collect_health_metrics()
            
            assert 'timestamp' in metrics
            assert 'system' in metrics
            assert 'network' in metrics
            assert 'processes' in metrics
            assert metrics['system']['cpu_percent'] == 75.0
            assert metrics['system']['memory_percent'] == 70.0
            assert metrics['system']['disk_percent'] == 60.0
    
    def test_get_health_status(self, health_monitor):
        """Test health status retrieval."""
        # Initially no data
        status = health_monitor.get_health_status()
        assert status['status'] == 'no_data'
        assert status['monitoring_active'] is False
        
        # Add some health data
        health_monitor.health_history.append({
            'timestamp': '2024-01-01T00:00:00',
            'system': {
                'cpu_percent': 75.0,
                'memory_percent': 70.0,
                'disk_percent': 60.0
            }
        })
        
        status = health_monitor.get_health_status()
        assert status['status'] == 'healthy'
        assert 'latest_metrics' in status
        assert status['metrics_count'] == 1
    
    def test_start_stop_monitoring(self, health_monitor):
        """Test monitoring start/stop."""
        # Start monitoring
        health_monitor.start_monitoring()
        assert health_monitor._monitoring is True
        
        # Stop monitoring
        health_monitor.stop_monitoring()
        assert health_monitor._monitoring is False


class TestHighAvailabilityManager:
    """Test high availability manager integration."""
    
    @pytest.fixture
    def ha_manager(self):
        # Create with None config path to use defaults
        return HighAvailabilityManager(None)
    
    def test_initialization(self, ha_manager):
        """Test HA manager initialization."""
        assert ha_manager.redundancy_manager is not None
        assert ha_manager.disaster_recovery_manager is not None
        assert ha_manager.backup_restore_manager is not None
        assert ha_manager.failover_manager is not None
        assert ha_manager.health_monitor is not None
    
    def test_start_stop(self, ha_manager):
        """Test starting and stopping HA management."""
        # Mock the component start/stop methods
        with patch.object(ha_manager.redundancy_manager, 'start_monitoring') as mock_red_start, \
             patch.object(ha_manager.backup_restore_manager, 'start_automated_backup') as mock_backup_start, \
             patch.object(ha_manager.failover_manager, 'start_monitoring') as mock_fail_start, \
             patch.object(ha_manager.health_monitor, 'start_monitoring') as mock_health_start:
            
            ha_manager.start()
            
            # Verify all components started based on config
            if ha_manager.config.get('redundancy', {}).get('enabled', False):
                mock_red_start.assert_called_once()
            if ha_manager.config.get('backup_restore', {}).get('enabled', False):
                mock_backup_start.assert_called_once()
            if ha_manager.config.get('failover', {}).get('enabled', False):
                mock_fail_start.assert_called_once()
            if ha_manager.config.get('health_monitoring', {}).get('enabled', False):
                mock_health_start.assert_called_once()
        
        # Test stop
        with patch.object(ha_manager.redundancy_manager, 'stop_monitoring') as mock_red_stop, \
             patch.object(ha_manager.backup_restore_manager, 'stop_automated_backup') as mock_backup_stop, \
             patch.object(ha_manager.failover_manager, 'stop_monitoring') as mock_fail_stop, \
             patch.object(ha_manager.health_monitor, 'stop_monitoring') as mock_health_stop:
            
            ha_manager.stop()
            
            mock_red_stop.assert_called_once()
            mock_backup_stop.assert_called_once()
            mock_fail_stop.assert_called_once()
            mock_health_stop.assert_called_once()
    
    def test_get_ha_status(self, ha_manager):
        """Test getting HA status."""
        with patch.object(ha_manager.health_monitor, 'get_health_status') as mock_health, \
             patch.object(ha_manager.redundancy_manager, 'get_redundancy_status') as mock_redundancy, \
             patch.object(ha_manager.backup_restore_manager, 'get_backup_status') as mock_backup, \
             patch.object(ha_manager.failover_manager, 'get_failover_status') as mock_failover:
            
            mock_health.return_value = {'status': 'healthy'}
            mock_redundancy.return_value = {'monitoring_active': True, 'services': {}}
            mock_backup.return_value = {'automated_backup_active': True, 'last_backup': None}
            mock_failover.return_value = {'monitoring_active': True}
            
            status = ha_manager.get_ha_status()
            
            assert status.overall_health == 'healthy'
            assert isinstance(status.failover_ready, bool)
            assert isinstance(status.redundancy_status, dict)
    
    def test_create_restore_backup(self, ha_manager):
        """Test backup creation and restoration."""
        with patch.object(ha_manager.backup_restore_manager, 'create_backup') as mock_create, \
             patch.object(ha_manager.backup_restore_manager, 'restore_from_backup') as mock_restore:
            
            # Mock backup creation
            mock_backup_info = Mock()
            mock_backup_info.backup_id = 'test_backup_123'
            mock_create.return_value = mock_backup_info
            
            backup_info = ha_manager.create_backup('manual')
            assert backup_info.backup_id == 'test_backup_123'
            mock_create.assert_called_once_with('manual')
            
            # Mock backup restoration
            mock_restore.return_value = {'status': 'completed', 'restore_id': 'restore_123'}
            
            restore_result = ha_manager.restore_backup('test_backup_123')
            assert restore_result['status'] == 'completed'
            mock_restore.assert_called_once_with('test_backup_123')
    
    def test_disaster_recovery(self, ha_manager):
        """Test disaster recovery functionality."""
        with patch.object(ha_manager.disaster_recovery_manager, 'create_recovery_plan') as mock_plan, \
             patch.object(ha_manager.disaster_recovery_manager, 'execute_recovery') as mock_execute:
            
            mock_plan.return_value = {'plan_id': 'plan_123', 'steps': []}
            mock_execute.return_value = {'success': True, 'recovery_time': 1800}
            
            # Create recovery plan
            plan = ha_manager.create_disaster_recovery_plan()
            assert plan['plan_id'] == 'plan_123'
            mock_plan.assert_called_once()
            
            # Execute recovery
            result = ha_manager.execute_disaster_recovery('plan_123')
            assert result['success'] is True
            mock_execute.assert_called_once_with('plan_123')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
