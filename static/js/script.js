// Allow Enter key to submit (guard if input exists)
const inputEl = document.getElementById('movieInput');
if (inputEl) {
    inputEl.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            getRecommendations();
        }
    });
}

// Footer year
document.addEventListener('DOMContentLoaded', () => {
    const y = document.getElementById('year');
    if (y) y.textContent = new Date().getFullYear();
});

// Suggestions (loaded on demand, titles hidden until search)
let allMoviesCache = null;
const suggestionsEl = document.getElementById('suggestions');

if (inputEl && suggestionsEl) {
inputEl.addEventListener('input', async function() {
    const query = inputEl.value.trim().toLowerCase();
    if (query.length < 2) {
        suggestionsEl.classList.remove('show');
        suggestionsEl.innerHTML = '';
        return;
    }

    try {
        if (!allMoviesCache) {
            const resp = await fetch('/movies');
            allMoviesCache = await resp.json();
        }

        const matches = allMoviesCache
            .filter(title => title && title.toLowerCase().includes(query))
            .slice(0, 8);

        if (matches.length === 0) {
            suggestionsEl.classList.remove('show');
            suggestionsEl.innerHTML = '';
            return;
        }

        suggestionsEl.innerHTML = matches
            .map(t => `<div class="suggestion-item" data-title="${t.replace(/"/g, '&quot;')}">${t}</div>`)
            .join('');
        suggestionsEl.classList.add('show');

        Array.from(suggestionsEl.children).forEach(item => {
            item.addEventListener('click', () => {
                const title = item.getAttribute('data-title');
                inputEl.value = title;
                suggestionsEl.classList.remove('show');
                suggestionsEl.innerHTML = '';
                getRecommendations();
            });
        });
    } catch (err) {
        console.error('Error loading suggestions:', err);
    }
});
}

document.addEventListener('click', (e) => {
    if (suggestionsEl && inputEl) {
        if (!suggestionsEl.contains(e.target) && e.target !== inputEl) {
            suggestionsEl.classList.remove('show');
        }
    }
});

async function getRecommendations() {
    const movieInput = document.getElementById('movieInput');
    const resultsDiv = document.getElementById('results');
    const recommendBtn = document.getElementById('recommendBtn');
    
    const title = movieInput.value.trim();
    
    if (!title) {
        alert('Please enter a movie title!');
        return;
    }

    // Show loading state
    recommendBtn.disabled = true;
    recommendBtn.textContent = 'Loading...';
    
    resultsDiv.innerHTML = '<div class="loading">Finding recommendations...</div>';
    resultsDiv.classList.add('show');

    try {
        const response = await fetch('/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ title: title })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to get recommendations');
        }

        displayResults(data.input_movie, data.recommendations);

    } catch (error) {
        console.error('Error:', error);
        resultsDiv.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${error.message}
                <br><br>
                <small>Try a different title or check availability.</small>
            </div>
        `;
    } finally {
        // Reset button state
        recommendBtn.disabled = false;
        recommendBtn.textContent = 'Search';
    }
}

function displayResults(inputMovie, recommendations) {
    const resultsDiv = document.getElementById('results');
    
    let html = `
        <h2>Recommendations</h2>
        <div class="input-movie">
            <strong>Based on:</strong> ${inputMovie}
        </div>
        <div class="recommendations-grid">
    `;

    recommendations.forEach((rec, index) => {
        const title = typeof rec === 'string' ? rec : rec.title;
        const reason = typeof rec === 'object' && rec.reason ? rec.reason : '';
        const similarity = typeof rec === 'object' && typeof rec.similarity === 'number' ? ` · similarity ${rec.similarity}` : '';
        const posterUrl = typeof rec === 'object' && rec.poster_url ? rec.poster_url : null;
        html += `
            <div class="movie-card">
                ${posterUrl ? `<img class="poster" src="${posterUrl}" alt="${title} poster" loading="lazy">` : ''}
                <h3>${index + 1}. ${title}</h3>
                ${reason ? `<div class=\"movie-meta\">${reason}${similarity}</div>` : ''}
            </div>
        `;
    });

    html += '</div>';
    resultsDiv.innerHTML = html;
}

