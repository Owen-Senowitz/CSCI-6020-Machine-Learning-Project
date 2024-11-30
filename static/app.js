document.getElementById("submit").addEventListener("click", async () => {
	const pickup_lat = document.getElementById("pickup_lat").value;
	const pickup_lon = document.getElementById("pickup_lon").value;
	const dropoff_lat = document.getElementById("dropoff_lat").value;
	const dropoff_lon = document.getElementById("dropoff_lon").value;

	if (!pickup_lat || !pickup_lon || !dropoff_lat || !dropoff_lon) {
		alert("Please enter all fields!");
		return;
	}

	const payload = {
		pickup_lat,
		pickup_lon,
		dropoff_lat,
		dropoff_lon,
	};

	try {
		const response = await fetch("/predict", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify(payload),
		});

		const result = await response.json();

		if (result.error) {
			document.getElementById("results").innerText = `Error: ${result.error}`;
		} else {
			const resultText = Object.entries(result)
				.map(
					([model, prediction]) => `${model}: ${prediction.toFixed(2)} seconds`
				)
				.join("\n");
			document.getElementById("results").innerText = resultText;
		}
	} catch (error) {
		document.getElementById("results").innerText = `Error: ${error.message}`;
	}
});
