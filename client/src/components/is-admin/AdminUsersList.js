export default function AdminUsersList({ userData, approveUser }) {
    return (
      <main className="max-w-2xl mx-auto p-6 text-black">
        {userData ? (
          <>
            {userData.map((user) => (
              <div key={user.id} className="flex">
                <section className="mb-6">
                  <h2 className="mb-2 text-center">
                    <strong>{user.u_first_name} {user.u_last_name}</strong>
                  </h2>
                  <div className="flex flex-col space-y-4">
                    <button  
                      className="bg-amber-900 text-white px-4 py-2 rounded-full hover:bg-amber-800"
                      onClick={() => approveUser(user)}
                    >
                      Approve User?
                    </button>
                  </div>
                </section>
              </div>
            ))}
          </>
        ) : null}
      </main>
    );
  }