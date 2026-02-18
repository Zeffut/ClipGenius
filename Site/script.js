/* ════════════════════════════════════════
   ClipGenius Landing — script.js
   ════════════════════════════════════════ */

/* ── Navbar scroll state ── */
const navbar = document.getElementById('navbar');
let ticking = false;

window.addEventListener('scroll', () => {
    if (!ticking) {
        requestAnimationFrame(() => {
            navbar.classList.toggle('scrolled', window.scrollY > 20);
            ticking = false;
        });
        ticking = true;
    }
}, { passive: true });

/* ── Scroll reveals (IntersectionObserver) ── */
const revealOpts = {
    root: null,
    rootMargin: '0px 0px -60px 0px',
    threshold: 0.08
};

const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        const el = entry.target;
        const delay = parseInt(el.dataset.delay || '0', 10);
        setTimeout(() => el.classList.add('visible'), delay);
        revealObserver.unobserve(el);
    });
}, revealOpts);

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

/* ── Animated progress ring in mockup ── */
const mockRing = document.getElementById('mockRing');
const mockPct  = document.getElementById('mockPct');

const RING_CIRCUMFERENCE = 238.76; // 2 * π * 38
const TARGET_PCT = 78;
const RING_DURATION = 2200; // ms

function animateRing() {
    if (!mockRing || !mockPct) return;

    let startTime = null;

    function easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    function step(timestamp) {
        if (!startTime) startTime = timestamp;
        const elapsed  = timestamp - startTime;
        const progress = Math.min(elapsed / RING_DURATION, 1);
        const eased    = easeOutCubic(progress);

        const currentPct = Math.round(eased * TARGET_PCT);
        const offset     = RING_CIRCUMFERENCE * (1 - (eased * TARGET_PCT / 100));

        mockRing.style.strokeDashoffset = offset;
        mockPct.childNodes[0].textContent = currentPct; // text node before <span>%

        if (progress < 1) {
            requestAnimationFrame(step);
        }
    }

    // Start after a short delay so it plays after page reveal
    setTimeout(() => requestAnimationFrame(step), 900);
}

// Observe the app window — start animation when it enters the viewport
const appWindowEl = document.getElementById('appWindow');
if (appWindowEl) {
    const ringObserver = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting) {
            animateRing();
            ringObserver.disconnect();
        }
    }, { threshold: 0.3 });
    ringObserver.observe(appWindowEl);
} else {
    animateRing();
}

/* ── Stat counter animation ── */
function animateCounter(el, target, duration) {
    const startTime = performance.now();
    function easeOutQuart(t) { return 1 - Math.pow(1 - t, 4); }

    function frame(now) {
        const elapsed  = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        el.textContent = Math.round(easeOutQuart(progress) * target);
        if (progress < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

const counterEls = document.querySelectorAll('.stat-num[data-count]');
if (counterEls.length) {
    const counterObs = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            const el     = entry.target;
            const target = parseInt(el.dataset.count, 10);
            animateCounter(el, target, 1800);
            counterObs.unobserve(el);
        });
    }, { threshold: 0.5 });

    counterEls.forEach(el => counterObs.observe(el));
}

/* ── Smooth-scroll for anchor links ── */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', e => {
        const href = anchor.getAttribute('href');
        if (href === '#') return;
        const target = document.querySelector(href);
        if (!target) return;
        e.preventDefault();
        const offset = 72; // nav height + buffer
        const top    = target.getBoundingClientRect().top + window.scrollY - offset;
        window.scrollTo({ top, behavior: 'smooth' });
    });
});

/* ── Hide scroll hint on first scroll ── */
const scrollHint = document.querySelector('.scroll-hint');
if (scrollHint) {
    const hideHint = () => {
        scrollHint.style.opacity = '0';
        scrollHint.style.transition = 'opacity .5s';
        window.removeEventListener('scroll', hideHint);
    };
    window.addEventListener('scroll', hideHint, { passive: true, once: true });
}

/* ── Seamless ticker (clone for loop) ── */
const tickerInner = document.querySelector('.ticker-inner');
if (tickerInner) {
    const clone = tickerInner.cloneNode(true);
    clone.setAttribute('aria-hidden', 'true');
    tickerInner.parentNode.appendChild(clone);
}

/* ── Accessibility: external links ── */
document.querySelectorAll('a[target="_blank"]').forEach(link => {
    if (!link.hasAttribute('rel')) link.setAttribute('rel', 'noopener noreferrer');
});
