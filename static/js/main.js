/* PACT project page: nav state, lazy video loading, autoplay, synced groups, copy. */
(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var hasIO = "IntersectionObserver" in window;

  /* ----- Sticky nav: show the bottom border after scrolling ----- */
  var nav = document.getElementById("nav");
  function updateNav() {
    nav.classList.toggle("is-scrolled", window.scrollY > 10);
  }
  window.addEventListener("scroll", updateNav, { passive: true });
  updateNav();

  /* ----- Placeholder links (arXiv, author pages) do nothing yet ----- */
  document.querySelectorAll('[data-placeholder="true"]').forEach(function (a) {
    a.addEventListener("click", function (e) { e.preventDefault(); });
  });

  /* ----- Lazy source loading: set src from data-src when near the viewport ----- */
  function ensureSrc(video) {
    var src = video.getAttribute("data-src");
    if (src) {
      video.removeAttribute("data-src");
      video.src = src;
    }
  }

  var lazyVideos = document.querySelectorAll("video[data-src]");
  if (hasIO) {
    var loader = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          ensureSrc(entry.target);
          loader.unobserve(entry.target);
        }
      });
    }, { rootMargin: "300px 0px" });
    lazyVideos.forEach(function (v) { loader.observe(v); });
  } else {
    lazyVideos.forEach(ensureSrc);
  }

  /* ----- Play helpers ----- */
  function play(video) {
    video.muted = true; // some browsers need the property, not only the attribute
    var p = video.play();
    return p && p.catch ? p.catch(function () {}) : Promise.resolve();
  }

  // Synced group (clips of equal length): the first video leads and the others follow it.
  function syncGroup(vids) {
    var lead = vids[0];
    vids.slice(1).forEach(function (v) {
      if (v.readyState >= 1 && Math.abs(v.currentTime - lead.currentTime) > 0.12) {
        v.currentTime = lead.currentTime;
      }
    });
  }

  function playGroup(stage) {
    var vids = Array.prototype.slice.call(stage.querySelectorAll("video"));
    vids.forEach(ensureSrc);
    if (!stage.hasAttribute("data-synced")) {
      stage.setAttribute("data-synced", "");
      // Correct drift a few times per second while the lead plays (skipped right after a loop).
      vids[0].addEventListener("timeupdate", function () {
        if (vids[0].currentTime > 0.3) syncGroup(vids);
      });
    }
    Promise.all(vids.map(play)).then(function () { syncGroup(vids); });
  }

  function pauseAll(el) {
    var vids = el.tagName === "VIDEO" ? [el] : el.querySelectorAll("video");
    Array.prototype.forEach.call(vids, function (v) { v.pause(); });
  }

  /* ----- Autoplay: play when at least 25% visible, pause otherwise ----- */
  var autoTargets = document.querySelectorAll("[data-autoplay], [data-autoplay-group]");

  if (reduceMotion) {
    // No autoplay: give clips native controls so they can be played on demand.
    document.querySelectorAll("[data-autoplay], [data-autoplay-group] video").forEach(function (v) {
      v.controls = true;
    });
  } else if (hasIO) {
    var player = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var el = entry.target;
        var visible = entry.isIntersecting && entry.intersectionRatio >= 0.25;
        if (visible) {
          if (el.hasAttribute("data-autoplay-group")) {
            playGroup(el);
          } else {
            ensureSrc(el);
            play(el);
          }
        } else {
          pauseAll(el);
        }
      });
    }, { threshold: [0, 0.25] });
    autoTargets.forEach(function (el) { player.observe(el); });
  }

  /* ----- Copy BibTeX ----- */
  var copyBtn = document.querySelector(".bibtex__copy");
  var code = document.getElementById("bibtex-code");
  if (copyBtn && code) {
    var resetTimer;
    copyBtn.addEventListener("click", function () {
      var text = code.textContent;
      var done = function () {
        copyBtn.textContent = "Copied";
        copyBtn.classList.add("is-copied");
        clearTimeout(resetTimer);
        resetTimer = setTimeout(function () {
          copyBtn.textContent = "Copy";
          copyBtn.classList.remove("is-copied");
        }, 1500);
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done, fallbackCopy);
      } else {
        fallbackCopy();
      }
      // Fallback for insecure contexts (e.g. file://): select and copy.
      function fallbackCopy() {
        var range = document.createRange();
        range.selectNodeContents(code);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        try { if (document.execCommand("copy")) done(); } catch (e) { /* leave selected */ }
      }
    });
  }
})();
