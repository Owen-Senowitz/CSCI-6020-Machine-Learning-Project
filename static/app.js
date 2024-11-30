document.getElementById("submit").addEventListener("click", async () => {
	const pickup_datetime = document.getElementById("pickup_datetime").value;
	const pickup_longitude = document.getElementById("pickup_longitude").value;
	const pickup_latitude = document.getElementById("pickup_latitude").value;
	const dropoff_longitude = document.getElementById("dropoff_longitude").value;
	const dropoff_latitude = document.getElementById("dropoff_latitude").value;

	if (
		!pickup_datetime ||
		!pickup_longitude ||
		!pickup_latitude ||
		!dropoff_longitude ||
		!dropoff_latitude
	) {
		alert("Please fill in all fields!");
		return;
	}

	const payload = {
		pickup_datetime,
		pickup_longitude,
		pickup_latitude,
		dropoff_longitude,
		dropoff_latitude,
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

document.getElementById("fetch-results").addEventListener("click", async () => {
	try {
		const response = await fetch("/results");
		const results = await response.json();

		if (results.error) {
			document.getElementById(
				"evaluation-results"
			).innerText = `Error: ${results.error}`;
		} else {
			const resultsText = Object.entries(results)
				.map(
					([model, metrics]) =>
						`${model} - Mean Squared Error: ${metrics.mean_squared_error.toFixed(
							2
						)} seconds²`
				)
				.join("\n");
			document.getElementById("evaluation-results").innerText = resultsText;
		}
	} catch (error) {
		document.getElementById(
			"evaluation-results"
		).innerText = `Error: ${error.message}`;
	}
});
