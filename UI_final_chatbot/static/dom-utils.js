// DOM Utility functions to replace querySelectorAll usage
// Rule ID: 1020004 - Avoid using querySelectorAll

/**
 * Get element by ID safely
 * @param {string} id - Element ID
 * @returns {Element|null}
 */
function getElementSafe(id) {
    return document.getElementById(id);
}

/**
 * Get element by class name (first match only)
 * @param {string} className - Class name
 * @returns {Element|null}
 */
function getElementByClass(className) {
    const elements = document.getElementsByClassName(className);
    return elements.length > 0 ? elements[0] : null;
}

/**
 * Find first element by class name within parent
 * @param {string} className - Class name to search for
 * @param {Element} parent - Parent element to search within
 * @returns {Element|null}
 */
function findFirstByClass(className, parent = document) {
    const elements = parent.getElementsByClassName(className);
    return elements.length > 0 ? elements[0] : null;
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        getElementSafe,
        getElementByClass,
        findFirstByClass
    };
}