"use client";

import { useState } from "react";

export default function SignUp() {
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, showMessage] = useState(false);
  const [label, setLabel] = useState("");

  const userCreation = async () => {
    const formBody = JSON.stringify({
      u_first_name: firstName,
      u_last_name: lastName,
      u_email: email,
      u_password: password,
    });

    const res = await fetch("http://localhost:8080/user", {
      method: "POST",
      body: formBody,
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await res.json();
    if (res.ok || !(res.ok)) {
      showMessage(true);
    }

    setLabel(data.message);
  };

  const onSubmit = () => {
    userCreation();
  };

  return (
    <main className="flex flex-col justify-start items-center min-h-screen p-6">
      <div className="mt-4 sm:mx-auto sm:w-full sm:max-w-md bg-white p-6 rounded-lg shadow-lg">
        {message && (
          <label htmlFor="label" className="block text-center font-medium text-red-600">
            {label}
          </label>
        )}

        <div className="flex flex-col space-y-4">
          {/* First Name */}
          <div>
            <label htmlFor="firstname" className="block text-sm font-medium leading-6 text-gray-900">
              First Name
            </label>
            <div className="mt-2">
              <input
                id="firstname"
                name="firstname"
                type="text"
                required
                onChange={(e) => setFirstName(e.target.value)}
                className="form-input"
              />
            </div>
          </div>

          {/* Last Name */}
          <div>
            <label htmlFor="lastname" className="block text-sm font-medium leading-6 text-gray-900">
              Last Name
            </label>
            <div className="mt-2">
              <input
                id="lastname"
                name="lastname"
                type="text"
                required
                onChange={(e) => setLastName(e.target.value)}
                className="form-input"
              />
            </div>
          </div>

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

          {/* Sign-Up Button */}
          <div>
            <button
              className="btn-primary"
              onClick={onSubmit}
            >
              Sign Up
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}