"use client";
import { useState } from "react";
import { Popover } from "@headlessui/react";
import { Bars3Icon } from "@heroicons/react/24/outline";

export default function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="bg-blue-300 shadow-md"> 
      <nav
        className="mx-auto flex max-w-7xl items-center justify-between p-6 lg:px-8"
        aria-label="Global"
      >
        {/* Logo / Company Name */}
        <div className="flex lg:flex-1">
          <a href="/" className="-m-1.5 p-1.5 text-black text-xl font-semibold">
            <span className="sr-only">Course Advisor</span>
            Course Advisor
          </a>
        </div>

        {/* Mobile Menu Button */}
        <div className="flex lg:hidden">
          <button
            type="button"
            className="-m-2.5 inline-flex items-center justify-center rounded-md p-2.5 text-black hover:text-gray-700"
            onClick={() => setMobileMenuOpen(true)}
          >
            <span className="sr-only">Open main menu</span>
            <Bars3Icon className="h-6 w-6" aria-hidden="true" />
          </button>
        </div>

        {/* Navigation Links */}
        <Popover.Group className="hidden lg:flex lg:gap-x-12">
          <a href="/" className="text-sm font-semibold leading-6 text-black hover:text-gray-700 transition duration-200">
            Home
          </a>
          <a href="/signup" className="text-sm font-semibold leading-6 text-black hover:text-gray-700 transition duration-200">
            Sign-up
          </a>
          <a href="/login" className="text-sm font-semibold leading-6 text-black hover:text-gray-700 transition duration-200">
            Login
          </a>
        </Popover.Group>
      </nav>
    </header>
  );
}