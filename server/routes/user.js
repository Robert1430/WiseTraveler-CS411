const express = require("express");
const router = express.Router();
const database = require("../database");
const {hashPassword}=require("../utils/helper");

//user get API
router.get("/", (req, res) => {
  database.execute("select * from user", function (err, result) {
    res.send(result);
  });

  // res.send("Calling user get api")
});

//user post API
router.post("/", (req, res) => {
  try {

    const hashedPassword=hashPassword(req.body.u_password);

    // console.log(hashedPassword)

    database.execute(
      "insert into user (u_first_name,u_last_name,u_email,u_password,is_approved,is_admin) values (?,?,?,?,?,?)",
      [
        req.body.u_first_name,
        req.body.u_last_name,
        req.body.u_email,
        // req.body.u_password,
        hashedPassword,
        0,
        0,
      ],
      function (err, result) {
        //console.log("error: ", err);
        if (err) {
          res.status(500).send({
            status:500,
            message:"The email you that you entered is already in use. Please sign up using a different email.",
          });
        } else {
          res.status(200).send({
            status:200,
            message:"User successfully created! You are currently waiting to be approved by our admin"
          });
        }
      }
    );
  } catch (error) {
    console.log(error.message);
  }
});

//user put API -- update password by EMAIL
router.put("/:email", (req, res) => {
  try {
    const hashedPassword=hashPassword(req.body.u_password);
    // u_first_name | u_last_name | u_email        | u_password | is_approved | is_admin
    database.execute(
      "update user set u_password=? where u_email=?",
      [hashedPassword, req.params.email],
      function (err, result) {
        //console.log("password: ", req.body.u_password);

        if (result.affectedRows == 0) {
          res.status(401).send("Record not found");
        } else {
          res.status(200).send("Record updated successfully");
        }
      }
    );
  } catch (error) {
    console.log(error.message);
  }
});

//user put API -- update first and last name by ID
router.put("/id/:id", (req, res) => {
  try {
    // u_first_name | u_last_name | u_email        | u_password | is_approved | is_admin
    database.execute(
      "update user set u_first_name=?, u_last_name=? where u_id=?",
      [req.body.u_first_name, req.body.u_last_name, req.params.id],
      function (err, result) {
        if (result.affectedRows == 0) {
          res.status(401).send("Record not found");
        } else {
          res.status(200).send("Record updated successfully");
        }
      }
    );
  } catch (error) {
    console.log(error.message);
  }
});

//user get by email API
router.get("/:email", (req, res) => {
  try {
    database.execute(
      "SELECT * FROM user WHERE u_email = ?",
      [req.params.email],
      function (err, result) {
        if (err) {
          console.log(err);
          res.status(500).send("Error retrieving user by email");
        } else {
          if (result.length === 0) {
            res.status(404).send("User not found");
          } else {
            res.status(200).send(result);
          }
        }
      }
    );
  } catch (error) {
    console.log(error.message);
    res.status(500).send("Internal Server Error");
  }
});

//user getbyid API
router.delete("/:id", (req, res) => {
  try {
    // u_first_name | u_last_name | u_email        | u_password | is_approved | is_admin
    database.execute(
      "delete from user where u_id=?",
      [req.params.id],
      function (err, result) {
        if (result.affectedRows == 0) {
          res.status(401).send("Record not deleted");
        } else {
          res.status(200).send("Record deleted successfully!");
        }
      }
    );
  } catch (error) {
    console.log(error.message);
  }
});

//get users that were not approved
router.get("/approved/:approved", (req, res) => {
  database.execute(`select * from user where is_approved = 0`, function (err, result) {
    if (err) {
      res.status(500).send(err.message);
      return;
    }
    res.send(result);
  });
});

// update user to approved status
router.put("/approved/:id", (req, res) => {
  try {
    // u_first_name | u_last_name | u_email        | u_password | is_approved | is_admin
    database.execute(
      "update user set is_approved=? where u_id=?",
      [req.body.is_approved, req.params.id],
      function (err, result) {
        if (result.affectedRows == 0) {
          res.status(401).send("Record not found");
        } else {
          res.status(200).send("Record updated successfully");
        }
      }
    );
  } catch (error) {
    console.log(error.message);
  }
});

module.exports = router;