/**
 * firetv/remote_nav.js
 * ─────────────────────────────────────────────────────────────────
 * Amazon Fire TV 10-Foot UI Remote Control & D-Pad Navigation Handler
 * 
 * Maps standard Amazon Fire TV Voice Remote buttons:
 * - D-Pad Up (38) / Down (40) / Left (37) / Right (39)
 * - Select / Enter (13)
 * - Back (27 / 8 / 10009 on Fire OS)
 * - Play / Pause (179)
 * - Fast-Forward / Rewind (228 / 227)
 * 
 * Provides:
 * 1. Spatial navigation with high-visibility glowing TV focus rings
 * 2. Automatic overscan padding safe-zone compliance
 * 3. Audio chime playback on high-volatility events
 */

(function() {
  'use strict';

  const FIRETV_KEYS = {
    SELECT: 13,
    LEFT: 37,
    UP: 38,
    RIGHT: 39,
    DOWN: 40,
    BACK: 27,
    BACK_FIREOS: 8,
    BACK_SAMSUNG: 10009,
    PLAY_PAUSE: 179
  };

  let currentFocusIndex = 0;
  let focusableElements = [];

  function getFocusableElements() {
    return Array.from(document.querySelectorAll(
      'button, [tabindex="0"], input, select, .stButton > button, a[href], .tv-card-interactive'
    )).filter(el => {
      const style = window.getComputedStyle(el);
      return style.display !== 'none' && style.visibility !== 'hidden' && !el.disabled;
    });
  }

  function updateFocus(newIndex) {
    focusableElements = getFocusableElements();
    if (!focusableElements.length) return;

    if (newIndex < 0) newIndex = 0;
    if (newIndex >= focusableElements.length) newIndex = focusableElements.length - 1;

    // Remove focus class from previously focused element
    if (focusableElements[currentFocusIndex]) {
      focusableElements[currentFocusIndex].classList.remove('tv-focused');
      focusableElements[currentFocusIndex].blur();
    }

    currentFocusIndex = newIndex;
    const target = focusableElements[currentFocusIndex];
    if (target) {
      target.classList.add('tv-focused');
      target.focus();
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  function handleKeyDown(e) {
    focusableElements = getFocusableElements();
    const keyCode = e.keyCode || e.which;

    switch (keyCode) {
      case FIRETV_KEYS.UP:
        e.preventDefault();
        updateFocus(currentFocusIndex - 1);
        break;

      case FIRETV_KEYS.DOWN:
        e.preventDefault();
        updateFocus(currentFocusIndex + 1);
        break;

      case FIRETV_KEYS.LEFT:
        e.preventDefault();
        updateFocus(currentFocusIndex - 1);
        break;

      case FIRETV_KEYS.RIGHT:
        e.preventDefault();
        updateFocus(currentFocusIndex + 1);
        break;

      case FIRETV_KEYS.SELECT:
        if (focusableElements[currentFocusIndex]) {
          focusableElements[currentFocusIndex].click();
        }
        break;

      case FIRETV_KEYS.BACK:
      case FIRETV_KEYS.BACK_FIREOS:
      case FIRETV_KEYS.BACK_SAMSUNG:
        // Handle Back navigation without closing webview if in modal
        const closeBtn = document.querySelector('.modal-close, button[data-testid="modal-close"]');
        if (closeBtn) {
          e.preventDefault();
          closeBtn.click();
        }
        break;

      default:
        break;
    }
  }

  // Inject TV Focus styles
  function injectTvStyles() {
    const style = document.createElement('style');
    style.id = 'firetv-nav-styles';
    style.innerHTML = `
      body {
        margin: 0;
        padding: 4vh 5vw !important; /* TV Safe Area (Overscan margin) */
        background-color: #020617 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      }
      .tv-focused {
        outline: 4px solid #38bdf8 !important;
        outline-offset: 4px !important;
        box-shadow: 0 0 24px rgba(56, 189, 248, 0.6) !important;
        transform: scale(1.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
      }
      /* TV Large typography */
      .tv-text-header {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
      }
      .tv-text-metric {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
      }
    `;
    document.head.appendChild(style);
  }

  // Initialize
  document.addEventListener('DOMContentLoaded', () => {
    injectTvStyles();
    document.addEventListener('keydown', handleKeyDown);
    setTimeout(() => {
      updateFocus(0);
    }, 500);
  });

  window.MacroPulseFireTV = {
    updateFocus,
    triggerVoiceDisplay: function(targetView) {
      console.log("Fire TV Voice Display Target:", targetView);
      const sel = document.querySelector(`[data-tab="${targetView}"]`);
      if (sel) sel.click();
    }
  };
})();
