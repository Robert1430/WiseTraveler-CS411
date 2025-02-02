"use client";

import { useState } from "react";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, showMessage] = useState(false);
  const [label, setLabel] = useState("");
  const [otp, setOtp] = useState("12345"); // Default OTP

  const userLogin = async () => {
    const formBody = JSON.stringify({
      u_email: email,
      u_password: password,
    });

    const res = await fetch("http://localhost:8080/login", {
      method: "POST",
      body: formBody,
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await res.json();
    if (res.ok) {
      showMessage(true);
      if (data.data.verify) {
        setOtp(data.data.verify); // Set OTP if received from backend
      }
      alert("Login successful! Redirecting...");
      window.location.href = `/profile?email=${email}`;
    } else {
      showMessage(true);
    }

    setLabel(data.message);
  };

  const onSubmit = () => {
    userLogin();
  };

  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <div className="mt-4 sm:mx-auto sm:w-full sm:max-w-md bg-white p-6 rounded-lg shadow-lg">
        {message && (
          <label htmlFor="label" className="block text-center font-medium text-red-600">
            {label}
          </label>
        )}

        <div className="flex flex-col space-y-4">
          {/* Email */}
          <div>
            <label htmlFor="email" className="block text-sm font-medium leading-6 text-gray-900">
              Email address
            </label>
            <div className="mt-2">
              <input
                id="email"
                name="email"
                type="email"
                autoComplete="email"
                required
                onChange={(e) => setEmail(e.target.value)}
                className="form-input"
              />
            </div>
          </div>

          {/* Password */}
          <div>
            <label htmlFor="password" className="block text-sm font-medium leading-6 text-gray-900">
              Password
            </label>
            <div className="mt-2">
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                onChange={(e) => setPassword(e.target.value)}
                className="form-input"
              />
            </div>
          </div>

          {/* OTP Field (Pre-filled with default OTP) */}
          <div>
            <label htmlFor="otp" className="block text-sm font-medium leading-6 text-gray-900">
              One-Time Password (OTP)
            </label>
            <div className="mt-2">
              <input
                id="otp"
                name="otp"
                type="text"
                value={otp}
                className="form-input"
                readOnly
              />
            </div>
          </div>

          {/* Login Button */}
          <div>
            <button className="btn-primary" onClick={onSubmit}>
              Login
            </button>
          </div>

          <a className="text-blue-700 text-center hover:underline" href="/forgot-password">
            Forgot Password?
          </a>
        </div>
      </div>
    </main>
  );
}