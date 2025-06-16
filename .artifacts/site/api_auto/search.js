
        let searchIndex = [];

        // Load search index
        fetch('search_index.json')
            .then(response => response.json())
            .then(data => {
                searchIndex = data;
            });

        // Search functionality
        document.getElementById('search-input').addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase();
            const results = document.getElementById('search-results');

            if (query.length < 2) {
                results.innerHTML = '';
                return;
            }

            const matches = searchIndex.filter(item =>
                item.name.toLowerCase().includes(query) ||
                item.module.toLowerCase().includes(query) ||
                item.description.toLowerCase().includes(query)
            ).slice(0, 10);

            results.innerHTML = matches.map(item => `
                <div class="search-result" onclick="window.location.href='${item.url}'">
                    <h4>${item.type === 'function' ? '🔧' : '🏗️'} ${item.name}</h4>
                    <p><strong>${item.module}</strong></p>
                    <p>${item.description}</p>
                </div>
            `).join('');
        });
