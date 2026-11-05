document.addEventListener('DOMContentLoaded', () => {
  const input = document.querySelector('.search-form input');
  if (!input) return;

  input.addEventListener('focus', () => {
    input.placeholder = 'Search by movie title...';
  });

  input.addEventListener('blur', () => {
    input.placeholder = 'Try: Inception, The Dark Knight, Titanic';
  });
});
