const form = document.getElementById("sentiment-form");
const textInput = document.getElementById("text-input");
const resultContainer = document.getElementById("result-container");
const errorContainer = document.getElementById("error-container");
const spinner = document.getElementById("spinner");
const analyzeBtn = document.getElementById("analyze-btn");
const btnText = document.getElementById("btn-text");
const analyzeIcon = document.getElementById("analyze-icon");
const tabText = document.getElementById("tab-text");
const tabSocial = document.getElementById("tab-social");
const textSection = document.getElementById("text-analysis-section");
const socialSection = document.getElementById("social-analysis-section");
const socialForm = document.getElementById("social-form");
const socialUrlInput = document.getElementById("social-url-input");
const analyzeSocialBtn = document.getElementById("analyze-social-btn");
const socialBtnText = document.getElementById("social-btn-text");
const spinnerSocial = document.getElementById("spinner-social");
const analyzeSocialIcon = document.getElementById("analyze-social-icon");
const socialResultContainer = document.getElementById(
  "social-result-container"
);
let sentimentChart = null;

function switchTab(activeTab) {
  resultContainer.classList.add("hidden");
  socialResultContainer.classList.add("hidden");
  errorContainer.classList.add("hidden");

  if (activeTab === "text") {
    textSection.classList.remove("hidden");
    socialSection.classList.add("hidden");

    tabText.classList.add("bg-indigo-600", "text-white");
    tabText.classList.remove(
      "text-gray-500",
      "hover:text-gray-700",
      "hover:bg-gray-100"
    );

    tabSocial.classList.remove("bg-indigo-600", "text-white");
    tabSocial.classList.add(
      "text-gray-500",
      "hover:text-gray-700",
      "hover:bg-gray-100"
    );
  } else {
    socialSection.classList.remove("hidden");
    textSection.classList.add("hidden");

    tabSocial.classList.add("bg-indigo-600", "text-white");
    tabSocial.classList.remove(
      "text-gray-500",
      "hover:text-gray-700",
      "hover:bg-gray-100"
    );

    tabText.classList.remove("bg-indigo-600", "text-white");
    tabText.classList.add(
      "text-gray-500",
      "hover:text-gray-700",
      "hover:bg-gray-100"
    );
  }
}

tabText.addEventListener("click", () => switchTab("text"));
tabSocial.addEventListener("click", () => switchTab("social"));

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = textInput.value;
  if (!text.trim()) {
    showError("Vui lòng nhập nội dung để phân tích.");
    return;
  }
  socialResultContainer.classList.add("hidden");
  resultContainer.classList.add("hidden");
  errorContainer.classList.add("hidden");
  spinner.classList.remove("hidden");
  analyzeIcon.classList.add("hidden");
  btnText.textContent = "Đang phân tích...";
  analyzeBtn.disabled = true;

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();
    if (!response.ok) {
      showError(data.error || "Đã xảy ra lỗi không xác định.");
    } else {
      displayResult(data);
    }
  } catch (error) {
    console.log(error);
    showError("Không thể kết nối đến máy chủ. Vui lòng kiểm tra lại.");
  } finally {
    spinner.classList.add("hidden");
    analyzeIcon.classList.remove("hidden");
    btnText.textContent = "Phân tích Văn bản";
    analyzeBtn.disabled = false;
  }
});

textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.ctrlKey) {
    e.preventDefault();
    analyzeBtn.click();
  }
});

socialForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const url = socialUrlInput.value;

  try {
    new URL(url);
  } catch (_) {
    showError("Vui lòng nhập một đường link (URL) hợp lệ.");
    return;
  }

  resultContainer.classList.add("hidden");
  socialResultContainer.classList.add("hidden");
  errorContainer.classList.add("hidden");
  spinnerSocial.classList.remove("hidden");
  analyzeSocialIcon.classList.add("hidden");
  socialBtnText.textContent = "Đang xử lý...";
  analyzeSocialBtn.disabled = true;

  try {
    const response = await fetch("/analyze-url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await response.json();
    if (!response.ok) {
      showError(data.error || "Lỗi không xác định khi phân tích link.");
    } else {
      displaySocialResult(data);
    }
  } catch (error) {
    showError("Không thể kết nối đến máy chủ để phân tích link.");
  } finally {
    spinnerSocial.classList.add("hidden");
    analyzeSocialIcon.classList.remove("hidden");
    socialBtnText.textContent = "Phân tích Link";
    analyzeSocialBtn.disabled = false;
  }
});

function displayResult(data) {
  const resultCard = document.getElementById("result-card");
  document.getElementById("result-emoji").textContent = data.emoji;
  document.getElementById("result-sentiment").textContent = data.sentiment;
  document.getElementById(
    "result-language"
  ).textContent = `(Ngôn ngữ: ${data.language})`;
  document.getElementById("result-score").textContent = `${data.score}%`;

  resultCard.className = "p-6 rounded-lg border-2 transition-all duration-300";
  resultCard.classList.add(...data.color.split(" "));

  const scoreCircle = document.getElementById("score-circle");
  scoreCircle.style.transition = "none";
  scoreCircle.style.strokeDasharray = "0, 100";

  setTimeout(() => {
    scoreCircle.style.transition = "stroke-dasharray 0.8s ease-in-out";
    document.getElementById(
      "score-circle"
    ).style.strokeDasharray = `${data.score}, 100`;
  }, 100);

  const emotionsList = document.getElementById("emotions-list");
  emotionsList.innerHTML = "";
  if (data.emotions && data.emotions.length > 0) {
    data.emotions.forEach((emotion) => {
      emotionsList.innerHTML += `
                        <div class="flex items-center bg-gray-100 text-gray-800 text-sm font-medium px-3 py-1 rounded-full">
                            <span class="mr-2">${emotion.emoji}</span>
                            <span>${emotion.label}</span>
                            <span class="text-gray-500 ml-2">${emotion.score}%</span>
                        </div>
                    `;
    });
    document.getElementById("emotions-container").classList.remove("hidden");
  } else {
    document.getElementById("emotions-container").classList.add("hidden");
  }

  resultContainer.classList.remove("hidden", "fade-in");
  void resultContainer.offsetWidth;
  resultContainer.classList.add("fade-in");
}

function displaySocialResult(data) {
  document.getElementById(
    "positive-percent"
  ).textContent = `${data.positive_percent}%`;
  document.getElementById(
    "negative-percent"
  ).textContent = `${data.negative_percent}%`;
  document.getElementById("total-comments").textContent = data.total_comments;
  const ctx = document.getElementById("sentimentChart").getContext("2d");
  if (sentimentChart) {
    sentimentChart.destroy();
  }
  sentimentChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Tích cực", "Tiêu cực"],
      datasets: [
        {
          data: [data.positive_percent, data.negative_percent],
          backgroundColor: ["rgb(34, 197, 94)", "rgb(239, 68, 68)"],
          hoverOffset: 4,
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      cutout: "60%",
      plugins: { legend: { position: "bottom" } },
    },
  });

  const commentListContainer = document.getElementById("comment-details-list");
  const commentDetailsSection = document.getElementById(
    "comment-details-section"
  );

  commentListContainer.innerHTML = "";

  if (data.details && data.details.length > 0) {
    data.details.forEach((comment) => {
      let sentimentClass = "";
      let sentimentIcon = "";

      switch (comment.sentiment) {
        case "Tích cực":
          sentimentClass = "bg-green-100 text-green-800";
          sentimentIcon = "👍";
          break;
        default:
          sentimentClass = "bg-red-100 text-red-800";
          sentimentIcon = "👎";
          break;
      }

      const commentElementHTML = `
                        <div class="border border-gray-200 rounded-lg p-3 bg-white">
                            <p class="text-gray-700 text-sm">"${comment.text}"</p>
                            <div class="flex justify-end items-center mt-2">
                                <span class="text-xs font-medium px-2.5 py-0.5 rounded-full ${sentimentClass}">
                                    ${sentimentIcon} ${comment.sentiment} (${comment.score}%)
                                </span>
                            </div>
                        </div>
                    `;
      commentListContainer.innerHTML += commentElementHTML;
    });
    commentDetailsSection.classList.remove("hidden");
  } else {
    commentDetailsSection.classList.add("hidden");
  }

  socialResultContainer.classList.remove("hidden");
}

function showError(message) {
  const errorMessage = document.getElementById("error-message");
  errorMessage.textContent = message;
  resultContainer.classList.add("hidden");
  socialResultContainer.classList.add("hidden");

  errorContainer.classList.remove("hidden", "fade-in");
  void errorContainer.offsetWidth;
  errorContainer.classList.add("fade-in");
}
