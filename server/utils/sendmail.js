const nodemailer = require("nodemailer");

async function sendEmail(email, mailSubject, body) {
  try {
    if (!process.env.SMTP_MAIL || !process.env.SMTP_PASSWORD) {
      throw new Error("SMTP credentials are missing in environment variables.");
    }

    const transporter = nodemailer.createTransport({
      host: "smtp.gmail.com",
      port: 587,
      secure: false, // True for 465, false for 587
      requireTLS: true,
      auth: {
        user: process.env.SMTP_MAIL,
        pass: process.env.SMTP_PASSWORD,
      },
    });

    const mailOptions = {
      from: process.env.SMTP_MAIL,
      to: email,
      subject: mailSubject,
      html: body,
    };

    const result = await transporter.sendMail(mailOptions);
    console.log(`✅ Email sent successfully to ${email}: ${result.response}`);
    
    return { success: true, response: result.response }; // ✅ Return structured response

  } catch (error) {
    console.error(`❌ Error sending email to ${email}:`, error);
    return { success: false, error: error.message }; // ✅ Always return an object
  }
}

module.exports = { sendEmail };