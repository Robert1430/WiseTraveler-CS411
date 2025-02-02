"use client";
import Login from "@/components/login/login";

export default function LoginPage() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-gray-100 p-6">
      <div className="auth-box">
        <div className="sm:mx-auto sm:w-full sm:max-w-sm">
          <h2 className="mt-5 text-center text-2xl font-bold leading-9 tracking-tight text-gray-900">
            Log Into Your Course Portal
          </h2>
        </div>

        <Login />
      </div>
    </main>
  );
}