// Generic helper definitions for shared library support
#if defined _WIN32 || defined __CYGWIN__
    #define PI_HELPER_DLL_IMPORT __declspec(dllimport)
    #define PI_HELPER_DLL_EXPORT __declspec(dllexport)
    #define PI_HELPER_DLL_LOCAL
#else
    #if __GNUC__ >= 4
        #define PI_HELPER_DLL_IMPORT __attribute__ ((visibility ("default")))
        #define PI_HELPER_DLL_EXPORT __attribute__ ((visibility ("default")))
        #define PI_HELPER_DLL_LOCAL  __attribute__ ((visibility ("hidden")))
    #else
        #define PI_HELPER_DLL_IMPORT
        #define PI_HELPER_DLL_EXPORT
        #define PI_HELPER_DLL_LOCAL
    #endif
#endif

// Now we use the generic helper definitions above to define PI_API and PI_LOCAL.
// PI_API is used for the public API symbols. It either DLL imports or DLL exports (or does nothing for static build)
// PI_LOCAL is used for non-api symbols.

#ifdef PI_DLL // defined if PI is compiled as a DLL
    #ifdef PI_DLL_EXPORTS // defined if we are building the PI DLL (instead of using it)
        #define PI_API PI_HELPER_DLL_EXPORT
    #else
        #define PI_API PI_HELPER_DLL_IMPORT
    #endif // PI_DLL_EXPORTS

    #define PI_LOCAL PI_HELPER_DLL_LOCAL
#else // PI_DLL is not defined: this means PI is a static lib.
    #define PI_API
    #define PI_LOCAL
#endif // PI_DLL
