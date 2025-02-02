import SignUp from "@/components/sign-up/sign-up";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
     <div className="flex min-h-full flex-1 flex-col justify-center px-6 lg:px-8">
        <div className="sm:mx-auto sm:w-full sm:max-w-sm">
          {/*           <img
            className="mx-auto h-10 w-auto"
            src="https://tailwindui.com/img/logos/mark.svg?color=indigo&shade=600"
            alt="Your Company"
          /> */}
          <h1 className="mt-5 text-center text-2xl font-bold leading-9 tracking-tight text-black">
          Welcome to the all-in-one Course Advising Platform
          </h1>
          <p className="mt-5 text-center text-xl leading-9 tracking-tight text-black">
            To begin, navigate to the header and either login or sign-up for a free account
          </p>
        </div>
      </div>
    </main>
  );
}
