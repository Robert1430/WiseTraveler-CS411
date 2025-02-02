import React, { useEffect, useState } from 'react';
import axios from 'axios';

export default function ResetPassword() {
  const [userEmail, setEmail] = useState('');
  const [password, setPassword] = useState("");

  const submitPassword = async () => {
      await changePassword();
      //alert("submit password works");
  };

  const changePassword = async () => {
    const formBody = JSON.stringify({
      u_password: password,
    });
   
   // alert("Change password works");

    try {
      const response = await axios.put(`http://localhost:8080/user/${userEmail}`, formBody, {
        headers: {
          "Content-Type": "application/json",
        },
      });
      console.log(response); 

      if(response.status == 200){
        alert("Your password was successfully reset");
      }

      else{
        alert("Error: your password was unable to be changed")
      }

    } catch (error) {
      console.error(error);
    }
}

  return (
    <main className="max-w-2xl mx-auto p-6 text-black">
        <div className="flex">
            <section className="mb-6">
                <div className="flex flex-col space-y-4">
                    
                    {/* email */}
                    <label htmlFor="current-password" className="block">Email Address</label>
                    <input 
                    type="email" 
                    id="current-email" 
                    name="current-email" 
                    onChange={(e) => {
                      setEmail(e.target.value);
                    }}
                    className="border border-gray-300 rounded-full px-3 py-2" 
                    required/>
                    {/* end email */}
                    
                    {/* new password */}
                    <label htmlFor="new-password" className="block">New Password</label>
                    <input 
                    type="password" 
                    id="new-password" 
                    name="new-password" 
                    onChange={(e) => {
                      setPassword(e.target.value);
                    }}
                    className="border border-gray-300 rounded-full px-3 py-2" 
                    required/>
                    {/* end new password */}

                    <div>
                      <button  
                      className="flex w-full justify-center rounded-full bg-amber-900 px-3 py-1.5 text-sm font-semibold leading-6 text-white shadow-sm hover:bg-amber-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-900"
                      onClick={submitPassword}
                      >
                        Reset Password
                      </button>
                    </div>
                </div>
            </section>
        </div>
    </main>
  );
}