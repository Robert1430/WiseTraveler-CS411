"use client";
import Admin from "@/components/admin/Admin"; // Use the refactored Admin component

export default function AdminPage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <div className="flex min-h-full flex-1 flex-col justify-center px-6 lg:px-8">
        <div className="sm:mx-auto sm:w-full sm:max-w-sm">
          <h2 className="mt-10 text-center text-2xl font-bold leading-9 tracking-tight text-gray-900">
            Welcome Administrator
          </h2>
        </div>

        <Admin /> 
      </div>
    </main>
  );
}