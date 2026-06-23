"""
SecretManager 单元测试。
"""
import pytest
import os
from pathlib import Path
from src.utils import SecretManager, init_secret_manager, get_secret_manager


class TestSecretManager:
    """SecretManager 单元测试类。"""

    @pytest.fixture
    def secret_manager(self):
        """创建测试用 SecretManager 实例。"""
        return SecretManager(master_key="test_master_key_12345")

    def test_init_with_master_key(self):
        """测试使用主密钥初始化。"""
        sm = SecretManager(master_key="test_key")
        assert sm.master_key == "test_key"
        assert len(sm._secrets) == 0
        assert len(sm._audit_log) == 0

    def test_init_from_env(self):
        """测试从环境变量初始化。"""
        os.environ["SECRET_MASTER_KEY"] = "env_test_key"
        try:
            sm = SecretManager()
            assert sm.master_key == "env_test_key"
        finally:
            del os.environ["SECRET_MASTER_KEY"]

    def test_encrypt_decrypt_secret(self, secret_manager):
        """测试加密和解密。"""
        plaintext = "my_secret_api_key"
        encrypted = secret_manager.encrypt_secret(plaintext)
        
        # 加密后应该包含冒号分隔符
        assert ":" in encrypted
        
        # 解密应该得到原文
        decrypted = secret_manager.decrypt_secret(encrypted)
        assert decrypted == plaintext

    def test_encrypt_without_master_key(self):
        """测试无主密钥时加密失败。"""
        sm = SecretManager()
        sm.master_key = None
        
        with pytest.raises(ValueError, match="未设置主密钥"):
            sm.encrypt_secret("test")

    def test_decrypt_without_master_key(self):
        """测试无主密钥时解密失败。"""
        sm = SecretManager()
        sm.master_key = None
        
        with pytest.raises(ValueError, match="未设置主密钥"):
            sm.decrypt_secret("salt:ciphertext")

    def test_decrypt_invalid_format(self, secret_manager):
        """测试解密无效格式失败。"""
        with pytest.raises(ValueError, match="解密失败"):
            secret_manager.decrypt_secret("invalid_format")

    def test_set_secret_plaintext(self, secret_manager):
        """测试设置明文密钥。"""
        secret_manager.set_secret("api_key", "my_api_key")
        
        assert secret_manager._secrets["api_key"] == "my_api_key"
        assert secret_manager._secret_versions["api_key"] == 1
        assert len(secret_manager._audit_log) == 1
        assert secret_manager._audit_log[0]["action"] == "set"

    def test_set_secret_encrypted(self, secret_manager):
        """测试设置加密密钥。"""
        plaintext = "my_secret"
        encrypted = secret_manager.encrypt_secret(plaintext)
        
        secret_manager.set_secret("api_key", encrypted, encrypted=True)
        
        assert secret_manager._secrets["api_key"] == plaintext
        assert secret_manager._secret_versions["api_key"] == 1

    def test_get_secret(self, secret_manager):
        """测试获取密钥。"""
        secret_manager.set_secret("api_key", "my_api_key")
        
        value = secret_manager.get_secret("api_key")
        assert value == "my_api_key"
        
        # 应该有两条审计日志（set + get）
        assert len(secret_manager._audit_log) == 2
        assert secret_manager._audit_log[1]["action"] == "get"

    def test_get_secret_default(self, secret_manager):
        """测试获取不存在的密钥返回默认值。"""
        value = secret_manager.get_secret("nonexistent", default="default_value")
        assert value == "default_value"

    def test_rotate_secret(self, secret_manager):
        """测试密钥轮换。"""
        secret_manager.set_secret("api_key", "old_key")
        old_version = secret_manager._secret_versions["api_key"]
        
        secret_manager.rotate_secret("api_key", "new_key")
        
        assert secret_manager._secrets["api_key"] == "new_key"
        assert secret_manager._secret_versions["api_key"] == old_version + 1
        
        # 检查轮换审计日志
        rotate_logs = [log for log in secret_manager._audit_log if log["action"] == "rotate"]
        assert len(rotate_logs) == 1
        assert rotate_logs[0]["old_version"] == old_version
        assert rotate_logs[0]["new_version"] == old_version + 1

    def test_load_from_env(self, secret_manager):
        """测试从环境变量批量加载。"""
        os.environ["TEST_API_KEY"] = "test_key_value"
        os.environ["TEST_SECRET"] = "test_secret_value"
        
        try:
            mapping = {
                "TEST_API_KEY": "api_key",
                "TEST_SECRET": "secret",
            }
            secret_manager.load_from_env(mapping)
            
            assert secret_manager._secrets["api_key"] == "test_key_value"
            assert secret_manager._secrets["secret"] == "test_secret_value"
        finally:
            del os.environ["TEST_API_KEY"]
            del os.environ["TEST_SECRET"]

    def test_load_from_encrypted_file(self, secret_manager, tmp_path):
        """测试从加密文件加载。"""
        # 创建加密文件
        secrets = {
            "api_key": secret_manager.encrypt_secret("key1"),
            "db_password": secret_manager.encrypt_secret("password1"),
        }
        
        import yaml
        encrypted_file = tmp_path / "encrypted_secrets.yaml"
        with open(encrypted_file, "w") as f:
            yaml.dump(secrets, f)
        
        secret_manager.load_from_encrypted_file(str(encrypted_file))
        
        assert secret_manager._secrets["api_key"] == "key1"
        assert secret_manager._secrets["db_password"] == "password1"

    def test_load_from_nonexistent_file(self, secret_manager):
        """测试加载不存在的加密文件失败。"""
        with pytest.raises(FileNotFoundError):
            secret_manager.load_from_encrypted_file("/nonexistent/file.yaml")

    def test_audit_report(self, secret_manager):
        """测试审计报告。"""
        secret_manager.set_secret("key1", "value1")
        secret_manager.set_secret("key2", "value2")
        secret_manager.get_secret("key1")
        
        report = secret_manager.audit_report()
        
        assert report["total_secrets"] == 2
        assert "key1" in report["access_times"]
        assert "key2" in report["access_times"]
        assert report["versions"]["key1"] == 1
        assert report["versions"]["key2"] == 1
        assert len(report["recent_audit"]) == 3  # 2 set + 1 get

    def test_check_leak_risk_unused_secret(self, secret_manager):
        """测试检测长期未使用的密钥。"""
        import time
        secret_manager.set_secret("old_key", "value")
        # 模拟 91 天前的访问
        secret_manager._secret_access_times["old_key"] = time.time() - 91 * 86400
        
        risks = secret_manager.check_leak_risk()
        
        unused_risks = [r for r in risks if r["type"] == "unused_secret"]
        assert len(unused_risks) == 1
        assert unused_risks[0]["name"] == "old_key"
        assert unused_risks[0]["days_since_access"] > 90

    def test_check_leak_risk_frequent_rotation(self, secret_manager):
        """测试检测频繁轮换的密钥。"""
        # 模拟 11 次轮换
        for i in range(11):
            secret_manager.set_secret("frequent_key", f"value_{i}")
        
        risks = secret_manager.check_leak_risk()
        
        rotation_risks = [r for r in risks if r["type"] == "frequent_rotation"]
        assert len(rotation_risks) == 1
        assert rotation_risks[0]["name"] == "frequent_key"
        assert rotation_risks[0]["version_count"] == 11

    def test_audit_log_trim(self):
        """测试审计日志裁剪。"""
        sm = SecretManager(master_key="test_key", max_audit_log_size=5)
        
        # 添加 10 条记录
        for i in range(10):
            sm.set_secret(f"key_{i}", f"value_{i}")
        
        # 应该只保留最后 5 条
        assert len(sm._audit_log) == 5
        # 最早的应该是 key_5
        assert sm._audit_log[0]["name"] == "key_5"
        # 最新的应该是 key_9
        assert sm._audit_log[-1]["name"] == "key_9"


class TestGlobalSecretManager:
    """全局 SecretManager 测试。"""

    def test_init_secret_manager(self):
        """测试初始化全局 SecretManager。"""
        sm = init_secret_manager("test_master_key")
        assert sm.master_key == "test_master_key"
        assert get_secret_manager() is sm

    def test_get_secret_manager_auto_init(self):
        """测试自动初始化全局 SecretManager。"""
        # 重置全局实例
        import src.utils
        src.utils._secret_manager = None
        
        sm = get_secret_manager()
        assert sm is not None
        assert isinstance(sm, SecretManager)
