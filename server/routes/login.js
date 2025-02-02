const express = require("express");
const router = express.Router();
const database = require("../database");
const { comparePassword } = require("../utils/helper");
const jwt = require("jsonwebtoken");
const { sendEmail } = require("../utils/sendmail");

const DEFAULT_OTP = "12345";

router.post("/", async (req, res) => {
  try {
    database.execute(
      "SELECT * FROM user WHERE u_email=?",
      [req.body.u_email],
      async (err, result) => {
        if (err || result.length === 0) {
          return res.status(401).json({
            status: 401,
            message: "Invalid email or password. Please try again.",
          });
        }

        const user = result[0];

        if (!comparePassword(req.body.u_password, user.u_password)) {
          return res.status(401).json({
            status: 401,
            message: "Invalid email or password. Please try again.",
          });
        }

        const token = jwt.sign(
          { userId: user.u_id, email: req.body.u_email },
          process.env.TOKEN_SECRET_KEY,
          { expiresIn: "1h" }
        );

        const oneTimePassword = DEFAULT_OTP;

        res.status(200).json({
          data: {
            token: token,
            email: req.body.u_email,
            userId: user.u_id,
            verify: oneTimePassword,
            admin: user.is_admin,
          },
        });

        // Ensure sendEmail is awaited properly
        try {
          const emailResponse = await sendEmail(
            user.u_email,
            "Login Verification",
            `Your one-time password is ${oneTimePassword}`
          );

          if (emailResponse && emailResponse.success) {
            console.log(`✅ OTP sent to ${user.u_email}`);
          } else {
            console.error(`❌ Failed to send OTP to ${user.u_email}:`, emailResponse?.error);
          }
        } catch (error) {
          console.error("❌ Error sending OTP email:", error.message);
        }
      }
    );
  } catch (error) {
    console.error("❌ Server Error:", error.message);
    res.status(500).json({ status: 500, message: "Internal server error" });
  }
});

module.exports = router;