"""
tests/test_profile_and_reset.py
--------------------------------
Integration tests for:
- User registration & login
- Forgot password & token generation
- Password reset validation & single-use guarantee
- Profile name update & Change password
- Manual ML risk prediction API
"""

import hashlib
import unittest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from main import app
from models.database import SessionLocal
from models.user import User


class ProfileAndResetIntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from models.database import init_db
        init_db()
        cls.client = TestClient(app)
        cls.test_email = f"test_{int(datetime.utcnow().timestamp())}@example.com"
        cls.test_password = "SecurePassword123!"
        cls.new_password = "UpdatedPassword456@"

    def test_01_register_user(self):
        res = self.client.post("/register", json={
            "name": "Dr. Sarah Connor",
            "email": self.test_email,
            "password": self.test_password,
            "confirm_password": self.test_password,
        })
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertIn("access_token", data)
        self.assertEqual(data.get("token_type"), "bearer")

    def test_02_login_and_get_profile(self):
        res = self.client.post("/login", json={
            "email": self.test_email,
            "password": self.test_password,
        })
        self.assertEqual(res.status_code, 200)
        token = res.json()["access_token"]
        self.__class__.token = token

        # Fetch profile
        headers = {"Authorization": f"Bearer {token}"}
        prof_res = self.client.get("/profile", headers=headers)
        self.assertEqual(prof_res.status_code, 200)
        self.assertEqual(prof_res.json()["name"], "Dr. Sarah Connor")

    def test_03_update_profile_name(self):
        headers = {"Authorization": f"Bearer {self.token}"}
        res = self.client.put("/profile", json={"name": "Sarah Connor, MD"}, headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["name"], "Sarah Connor, MD")

    def test_04_change_password(self):
        headers = {"Authorization": f"Bearer {self.token}"}

        # Attempt with incorrect current password
        bad_res = self.client.post("/change-password", json={
            "current_password": "WrongPassword999!",
            "new_password": self.new_password,
            "confirm_password": self.new_password,
        }, headers=headers)
        self.assertEqual(bad_res.status_code, 400)

        # Successful change
        good_res = self.client.post("/change-password", json={
            "current_password": self.test_password,
            "new_password": self.new_password,
            "confirm_password": self.new_password,
        }, headers=headers)
        self.assertEqual(good_res.status_code, 200)

        # Verify login with new password succeeds
        login_res = self.client.post("/login", json={
            "email": self.test_email,
            "password": self.new_password,
        })
        self.assertEqual(login_res.status_code, 200)

    def test_05_forgot_password_and_reset(self):
        # 1. Request reset link
        forgot_res = self.client.post("/forgot-password", json={"email": self.test_email})
        self.assertEqual(forgot_res.status_code, 200)
        self.assertIn("message", forgot_res.json())

        # 2. Check token in DB
        db = SessionLocal()
        user = db.query(User).filter(User.email == self.test_email).first()
        self.assertIsNotNone(user.reset_token_hash)
        self.assertIsNotNone(user.reset_token_expires)

        # Create a known raw token and hash for test assertion
        raw_token = "unit_test_token_sample_12345"
        user.reset_token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        user.reset_token_expires = datetime.utcnow() + timedelta(minutes=15)
        db.commit()
        db.close()

        # 3. Reset password using the valid token
        brand_new_pw = "FinalBrandNewPass789!"
        reset_res = self.client.post("/reset-password", json={
            "token": raw_token,
            "new_password": brand_new_pw,
            "confirm_password": brand_new_pw,
        })
        self.assertEqual(reset_res.status_code, 200)

        # 4. Token must be invalidated immediately (single-use)
        reused_res = self.client.post("/reset-password", json={
            "token": raw_token,
            "new_password": brand_new_pw,
            "confirm_password": brand_new_pw,
        })
        self.assertEqual(reused_res.status_code, 400)

        # 5. Login works with the newly set password
        login_res = self.client.post("/login", json={
            "email": self.test_email,
            "password": brand_new_pw,
        })
        self.assertEqual(login_res.status_code, 200)
        new_token = login_res.json()["access_token"]

        # 6. Test manual prediction endpoint
        pred_res = self.client.post("/predict/manual", json={
            "glucose": 105.0,
            "hba1c": 5.6,
            "cholesterol": 190.0,
        }, headers={"Authorization": f"Bearer {new_token}"})
        self.assertEqual(pred_res.status_code, 200)
        pred_data = pred_res.json()
        self.assertIn("risk_label", pred_data)
        self.assertIn("confidence", pred_data)
        self.assertIn("extracted_params", pred_data)
        self.assertIn("disclaimer", pred_data)


if __name__ == "__main__":
    unittest.main()
