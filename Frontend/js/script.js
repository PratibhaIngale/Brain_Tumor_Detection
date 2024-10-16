document.getElementById('tumor-form').addEventListener('submit', async (event) => {
  event.preventDefault(); // Prevent default form submission

  const formData = new FormData();
  const imageUpload = document.getElementById('imageUpload');

  formData.append('image', imageUpload.files[0]); // Append the image file

  try {
      const response = await fetch('/api/detect-tumor', {
          method: 'POST',
          body: formData
      });

      const result = await response.json(); // Assuming the response is in JSON format

      if (response.ok) {
          document.getElementById('result').innerText = `Detection Result: ${result.message}`;
      } else {
          document.getElementById('result').innerText = `Error: ${result.message}`;
      }
  } catch (error) {
      document.getElementById('result').innerText = 'An error occurred while detecting the tumor.';
      console.error('Error:', error);
  }
});
