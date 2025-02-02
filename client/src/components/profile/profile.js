import React, { useEffect, useState } from 'react';
import axios from 'axios';

export default function Profile() {
  const [email, setEmail] = useState('');
  const [userData, setUserData] = useState(null);
  const [password, setPassword] = useState("");
  const [passwordConfirm, confirmPassword] = useState("");
  const [nowPassword, currentPassword] = useState("");
  const [firstName, changeFirstName] = useState("");
  const [lastName, changeLastName] = useState("");

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const userEmail = params.get('email');
    if (userEmail) {
      setEmail(userEmail);
      fetchData(userEmail);
    }
  }, []);

  const fetchData = async (email) => {
    try {
      const response = await axios.get(`http://localhost:8080/user/${email}`);
      setUserData(response.data[0]); 
    } catch (error) {
      console.error(error);
    }
  };

  const submitPassword = async () => {
    if(password == passwordConfirm){
        await changePassword();
        //alert("submit password works");
    }
    else{
      alert("The values you entered for your new password and confirm new password do not match");
    }
  };

  const submitName = async () => {
    await changeName();
  }

  const changeName = async () => {
    const formBody = JSON.stringify({
      u_first_name: firstName,
      u_last_name: lastName,
    })

    try {
      const response = await axios.put(`http://localhost:8080/user/id/${userData.u_id}`, formBody, {
        headers: {
          "Content-Type": "application/json",
        },
      });
      console.log(response);

      if(response.status == 200){
        alert("Your name was successfully changed");
        window.location.href = `http://localhost:3000/profile?email=${email}`;
      }

      else{
        alert("Error: your name was unable to be changed")
      }

    } catch (error) {
      console.error(error);
    }
  }

  const changePassword = async () => {
    const formBody = JSON.stringify({
      u_password: password,
    });
   
   // alert("Change password works");

    try {
      const response = await axios.put(`http://localhost:8080/user/${userData.u_email}`, formBody, {
        headers: {
          "Content-Type": "application/json",
        },
      });
      console.log(response);

      if(response.status == 200){
        alert("Your password was successfully changed");
        window.location.href = `http://localhost:3000/profile?email=${email}`;
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
        {userData ? (
        <>
        <section className="mb-6">
          <ul>
              <li className="mb-2 text-center"><strong>Email:</strong> {userData.u_email} </li>
              <li className="mb-2 text-center"><strong>Name:</strong> {userData.u_first_name} {userData.u_last_name} </li>
          </ul>
        </section>

        <div className="flex">
            <div className="w-1/2 mr-8">
                <section className="mb-6">
                    <h2 className="mb-2 text-center"><strong>Change Password?</strong></h2>
                    <div className="flex flex-col space-y-4">
                        
                        {/* current password */}
                        <label htmlFor="current-password" className="block">Current Password:</label>
                        <input 
                        type="password" 
                        id="current-password" 
                        name="current-password" 
                        onChange={(e) => {
                          currentPassword(e.target.value);
                        }}
                        className="border border-gray-300 rounded-full px-3 py-2" 
                        required/>
                        {/* end current password */}
                        
                        {/* new password */}
                        <label htmlFor="new-password" className="block">New Password:</label>
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

                        {/* confirm new password */}
                        <label htmlFor="confirm-password" className="block">Confirm New Password:</label>
                        <input
                         type="password" 
                         id="confirm-password" 
                         name="confirm-password" 
                         onChange={(e) => {
                          confirmPassword(e.target.value);
                        }}
                         className="border border-gray-300 rounded-full px-3 py-2" 
                         required/>
                         {/* end confirm new password */}

                        <button  
                        className="bg-amber-900 text-white px-4 py-2 rounded-full hover:bg-amber-800"
                        onClick={submitPassword}
                        >
                          Change Password
                        </button>
                    </div>
               </section>
            </div>

            <div className="w-1/2 mt-12">
                <section className="mb-6">
                    <h2 className="mb-2 text-center"><strong>Change Name?</strong></h2>
                    <div className="flex flex-col space-y-4">

                        {/* firstName */}
                        <label htmlFor="firstName" className="block">First Name:</label>
                        <input 
                        type="text" 
                        id="firstName" 
                        name="First Name" 
                        className="border border-gray-300 rounded-full px-3 py-2" 
                        required
                        onChange={(e) => {
                          changeFirstName(e.target.value);
                        }}/>
                        {/* end firstName */}

                        {/* lastName */}
                        <label htmlFor="lastName" className="block">Last Name:</label>
                        <input type="text" id="lastName" name="Last Name" 
                        className="border border-gray-300 rounded-full px-3 py-2"
                         required
                         onChange={(e) => {
                          changeLastName(e.target.value);
                         }}/>
                        {/* end lastName */}

                        <button type="submit" 
                        className="bg-amber-900 text-white px-4 py-2 rounded-full hover:bg-amber-800"
                        onClick={submitName}
                        >
                          Update Information
                        </button>
                    </div>
                </section>
            </div>

        </div>
        </>
        ) : null}
    </main>
  );
}