async function predictImage() {
    console.log("predictImage started");
    const input = document.getElementById('imageInput')
    const file = input.files[0];

    if (!file) {
        alert("Please select an image first!")
        return;
    }

    //show the image on the screen
    const preview = document.getElementById('previewImage')
    preview.src = URL.createObjectURL(file);

    //prepare the data for the api
    const formData = new FormData();
    formData.append("file", file);

    //send the data to the api
    try {
        const response = await fetch("/predict/", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        //Update the UI with the result
        document.getElementById('label').innerText = data.class.toUpperCase();

        //convert to the percentage
        const percentage = (data.confidence * 100).toFixed(2);
        document.getElementById('confidence').innerText = percentage;

        document.getElementById('result').classList.remove('hidden');
    } catch (error) {
        console.error("Error: ", error);
        alert("Something went wrong!");
    }
}