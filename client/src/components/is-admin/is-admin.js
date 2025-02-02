import React, { useEffect, useState } from "react";
import axios from "axios";
import AdminUsersList from "@/components/is-admin/AdminUsersList";

export default function Admin() {
  const [userData, setUserData] = useState(null);
  const [email, setEmail] = useState("");

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const userEmail = params.get("email");
    if (userEmail) {
      setEmail(userEmail);
      fetchData();
    }
  }, []);

  const fetchData = async () => {
    try {
      const response = await axios.get(`http://localhost:8080/user/approved/approved`);
      setUserData(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  const approveUser = async (user) => {
    const formBody = JSON.stringify({ is_approved: 1 });

    try {
      const res = await axios.put(`http://localhost:8080/user/approved/${user.u_id}`, formBody, {
        headers: { "Content-Type": "application/json" },
      });

      if (res.status === 200) {
        alert(`${user.u_first_name} ${user.u_last_name} was successfully approved`);
        fetchData(); // Refresh the data instead of reloading
      } else {
        alert("Error: User was unable to be approved");
      }
    } catch (error) {
      console.error(error);
    }
  };

  return <AdminUsersList userData={userData} approveUser={approveUser} />;
}