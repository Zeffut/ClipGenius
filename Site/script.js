// ==================== SMOOTH SCROLL ==================== */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        if (href === '#') return;

        e.preventDefault();
        const target = document.querySelector(href);
        if (target) {
            const offset = 60;
            const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - offset;
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
        }
    });
});

// ==================== NAVBAR ENHANCED ==================== */
let lastScrollTop = 0;
const navbar = document.querySelector('.navbar');

window.addEventListener('scroll', () => {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

    if (scrollTop > 50) {
        navbar.classList.add('scrolled');
        navbar.style.background = 'rgba(255, 255, 255, 0.98)';
    } else {
        navbar.classList.remove('scrolled');
        navbar.style.background = 'rgba(255, 255, 255, 0.95)';
    }

    lastScrollTop = scrollTop <= 0 ? 0 : scrollTop;
});

// ==================== INTERSECTION OBSERVER FOR ANIMATIONS ==================== */
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe all animated elements
document.querySelectorAll('.feature, .persona, .content-type, .problem-card, .solution-card, .step').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(10px)';
    el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(el);
});

// ==================== COUNTER ANIMATION ==================== */
function animateCounter(element, target) {
    let current = 0;
    const increment = target / 50;
    const duration = 2000;

    const counter = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = target;
            clearInterval(counter);
        } else {
            element.textContent = Math.floor(current);
        }
    }, duration / 50);
}

// Observe stats section
const statsObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const statNumbers = document.querySelectorAll('.stat-value');
            statNumbers.forEach(stat => {
                const text = stat.textContent.trim();
                if (text === '96%') {
                    animateCounter(stat, 96);
                    stat.textContent = '96%';
                }
            });
            statsObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

const heroStats = document.querySelector('.hero-stats');
if (heroStats) {
    statsObserver.observe(heroStats);
}

// ==================== BUTTON HOVER EFFECT ==================== */
document.querySelectorAll('.btn').forEach(btn => {
    btn.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-2px)';
    });

    btn.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0)';
    });
});

// ==================== CARD HOVER EFFECTS ==================== */
document.querySelectorAll('.feature, .persona, .content-type, .step').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.cursor = 'pointer';
    });
});

// ==================== LOAD ANIMATION ==================== */
window.addEventListener('load', () => {
    document.body.style.opacity = '1';
});

// ==================== ACTIVE NAV LINK ==================== */
const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('section[id]');

window.addEventListener('scroll', () => {
    let current = '';

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        if (pageYOffset >= sectionTop - 200) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.style.color = '';
        link.style.fontWeight = '';
        const href = link.getAttribute('href');
        if (href === '#' + current) {
            link.style.color = '#3b82f6';
            link.style.fontWeight = '600';
        }
    });
});

// ==================== ACCESSIBILITY ==================== */
document.querySelectorAll('a[target="_blank"]').forEach(link => {
    link.setAttribute('rel', 'noopener noreferrer');
});

// ==================== PARALLAX ON SCROLL ==================== */
const heroVisual = document.querySelector('.hero-visual');

if (heroVisual) {
    window.addEventListener('scroll', () => {
        const scrollPosition = window.scrollY;
        const heroSection = document.querySelector('.hero');
        const heroTop = heroSection.offsetTop;

        if (scrollPosition < heroTop + 500) {
            heroVisual.style.transform = `translateY(${scrollPosition * 0.5}px)`;
        }
    });
}

// ==================== CLIP CARDS STAGGER ANIMATION ==================== */
const clipCards = document.querySelectorAll('.clip-card');
clipCards.forEach((card, index) => {
    card.style.animation = `fadeIn 0.6s ease forwards ${index * 0.1}s`;
});

// ==================== PREFERS REDUCED MOTION ==================== */
if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    document.documentElement.style.scrollBehavior = 'auto';
    document.querySelectorAll('*').forEach(el => {
        el.style.animationDuration = '0.01ms';
        el.style.transitionDuration = '0.01ms';
    });
}

// ==================== RESPONSIVE BEHAVIOR ==================== */
function handleResponsive() {
    const isMobile = window.innerWidth <= 768;

    // Adjust animations based on device
    if (isMobile) {
        document.querySelectorAll('.clip-card').forEach(card => {
            card.style.animation = 'none';
        });
    }
}

window.addEventListener('resize', handleResponsive);
handleResponsive();
