import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Shield, 
  AlertTriangle, 
  BarChart3, 
  Users, 
  Globe, 
  Award,
  ArrowRight,
  CheckCircle,
  TrendingUp,
  Activity,
  Database,
  Lock,
  Sun,
  Moon,
  Sparkles
} from 'lucide-react';

const LandingPage = ({ onLogin }) => {
  const [isDarkMode, setIsDarkMode] = useState(false); // Light mode default
  const [mounted, setMounted] = useState(false);
  const [rollingText, setRollingText] = useState(0);
  const [typingText, setTypingText] = useState('');
  const [isTyping, setIsTyping] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setRollingText((prev) => (prev + 1) % 3);
    }, 3000); // Change every 3 seconds

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const currentWord = rollingWords[rollingText];
    let index = 0;
    setIsTyping(true);
    setTypingText('');

    // Type out the word
    const typeInterval = setInterval(() => {
      if (index < currentWord.length) {
        setTypingText(currentWord.slice(0, index + 1));
        index++;
      } else {
        clearInterval(typeInterval);
        // Wait a bit, then start deleting
        setTimeout(() => {
          // Backspace animation
          const backspaceInterval = setInterval(() => {
            setTypingText(prev => {
              if (prev.length > 0) {
                return prev.slice(0, -1);
              } else {
                clearInterval(backspaceInterval);
                setIsTyping(false);
                return '';
              }
            });
          }, 100); // Backspace speed
        }, 1500); // Wait longer before backspacing
      }
    }, 150); // Type speed

    return () => clearInterval(typeInterval);
  }, [rollingText]);

  const rollingWords = ['care', 'chain', 'support'];

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const navItems = [
    { id: 'home', label: 'Home' },
    { id: 'features', label: 'Features' },
    { id: 'stats', label: 'Stats' },
    { id: 'testimonials', label: 'Testimonials' },
    { id: 'cta', label: 'Get Started' }
  ];

  const stats = [
    { label: "Items Tracked", value: "1,389", icon: Database },
    { label: "Uptime", value: "99.9%", icon: Activity },
    { label: "Support", value: "24/7", icon: Users },
    { label: "Healthcare Partners", value: "89+", icon: Globe },
  ];

  const features = [
    {
      icon: Lock,
      title: "Advanced Security",
      description: "Every transaction is cryptographically secured and immutably recorded, ensuring complete transparency and audit trails."
    },
    {
      icon: AlertTriangle,
      title: "Real-time Alerts",
      description: "Instant notifications for low stock, expiring items, and critical inventory levels."
    },
    {
      icon: BarChart3,
      title: "Smart Analytics",
      description: "AI-powered insights and predictive analytics to optimize inventory management."
    },
    {
      icon: Users,
      title: "Multi-role Access",
      description: "Role-based access control for administrators, managers, staff, and suppliers."
    },
    {
      icon: Globe,
      title: "Global Compliance",
      description: "Meet international healthcare standards and regulatory requirements."
    },
    {
      icon: Award,
      title: "Proven Excellence",
      description: "Trusted by leading healthcare facilities worldwide for reliable inventory management."
    }
  ];

  const testimonials = [
    {
      quote: "MedCare has revolutionized our inventory management. The verification transparency gives us complete confidence in our supply chain.",
      author: "Dr. Sarah Johnson",
      role: "Chief Medical Officer",
      hospital: "City General Hospital"
    },
    {
      quote: "The automated alerts and real-time tracking have significantly reduced our stockouts and improved patient care.",
      author: "Michael Chen",
      role: "Hospital Administrator",
      hospital: "Regional Medical Center"
    },
    {
      quote: "The AI-powered analytics help us predict demand accurately and optimize our inventory levels like never before.",
      author: "Dr. Emily Rodriguez",
      role: "Director of Operations",
      hospital: "University Health System"
    }
  ];

  if (!mounted) return null;

  return (
    <div className={`min-h-screen transition-all duration-500 ${
      isDarkMode 
        ? 'bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white' 
        : 'bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 text-gray-900'
        } relative overflow-hidden`}>
      
      {/* Glass Panel Navigation */}
      <motion.nav 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, delay: 0.2 }}
        className={`fixed top-0 left-0 right-0 z-50 p-4 ${
          isDarkMode ? 'bg-white/10' : 'bg-white/20'
        } backdrop-blur-md border-b ${
          isDarkMode ? 'border-white/20' : 'border-gray-200/50'
        }`}
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          {/* Logo */}
          <motion.div 
            whileHover={{ scale: 1.05 }}
            className="text-2xl font-bold"
          >
            <span className={`bg-gradient-to-r from-blue-600 to-green-600 bg-clip-text text-transparent ${
              isDarkMode ? 'from-blue-400 to-green-400' : 'from-blue-600 to-green-600'
            }`}>
              Med
            </span>
            <span className={`bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent ${
              isDarkMode ? 'from-purple-400 to-pink-400' : 'from-purple-600 to-pink-600'
            }`}>
              Care
            </span>
          </motion.div>

          {/* Navigation Items */}
          <div className="hidden md:flex items-center space-x-8">
            {navItems.slice(0, -1).map((item) => (
              <motion.button
                key={item.id}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => scrollToSection(item.id)}
                className={`px-4 py-2 rounded-lg transition-all duration-300 shadow-sm ${
                  isDarkMode 
                    ? 'text-white/80 hover:text-white hover:bg-white/10 hover:shadow-lg' 
                    : 'text-gray-700 hover:text-gray-900 hover:bg-white/30 hover:shadow-lg'
                }`}
              >
                {item.label}
              </motion.button>
            ))}
            {/* Login Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onLogin}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-300 shadow-sm ${
                isDarkMode 
                  ? 'text-white/80 hover:text-white hover:bg-white/10 hover:shadow-lg' 
                  : 'text-gray-700 hover:text-gray-900 hover:bg-white/30 hover:shadow-lg'
              }`}
            >
              Login
            </motion.button>
            {/* Special Get Started Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => scrollToSection('cta')}
              className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-2 rounded-lg font-semibold shadow-lg hover:shadow-xl transition-all duration-300"
            >
              Get Started
            </motion.button>
          </div>

          {/* Mobile Menu Button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`md:hidden p-2 rounded-lg ${
              isDarkMode 
                ? 'text-white/80 hover:text-white hover:bg-white/10' 
                : 'text-gray-700 hover:text-gray-900 hover:bg-white/30'
            }`}
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </motion.button>
        </div>
      </motion.nav>
      
      {/* Smooth Moving Background */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Large blurred balls moving slowly */}
        <motion.div
          className={`absolute w-96 h-96 rounded-full blur-xl ${
            isDarkMode 
              ? 'bg-gradient-to-r from-blue-400/25 to-purple-400/25' 
              : 'bg-gradient-to-r from-blue-400/40 to-purple-400/40'
          }`}
          style={{
            top: '10%',
            left: '10%',
          }}
          animate={{
            x: [0, 100, 0],
            y: [0, -50, 0],
            scale: [1, 1.1, 1],
          }}
          transition={{
            duration: 30,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        
        <motion.div
          className={`absolute w-80 h-80 rounded-full blur-xl ${
            isDarkMode 
              ? 'bg-gradient-to-r from-purple-400/25 to-pink-400/25' 
              : 'bg-gradient-to-r from-purple-400/40 to-pink-400/40'
          }`}
          style={{
            top: '60%',
            right: '15%',
          }}
          animate={{
            x: [0, -80, 0],
            y: [0, 40, 0],
            scale: [1, 1.2, 1],
          }}
          transition={{
            duration: 35,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        
        <motion.div
          className={`absolute w-72 h-72 rounded-full blur-xl ${
            isDarkMode 
              ? 'bg-gradient-to-r from-green-400/20 to-blue-400/20' 
              : 'bg-gradient-to-r from-green-400/35 to-blue-400/35'
          }`}
          style={{
            top: '30%',
            right: '30%',
          }}
          animate={{
            x: [0, 60, 0],
            y: [0, -30, 0],
            rotate: [0, 180, 360],
          }}
          transition={{
            duration: 40,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* More moving balls for richer background */}
        <motion.div
          className={`absolute w-64 h-64 rounded-full blur-lg ${
            isDarkMode 
              ? 'bg-gradient-to-r from-indigo-400/20 to-cyan-400/20' 
              : 'bg-gradient-to-r from-indigo-400/35 to-cyan-400/35'
          }`}
          style={{
            top: '70%',
            left: '5%',
          }}
          animate={{
            x: [0, 120, 0],
            y: [0, -40, 0],
            scale: [1, 1.15, 1],
          }}
          transition={{
            duration: 45,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        <motion.div
          className={`absolute w-56 h-56 rounded-full blur-lg ${
            isDarkMode 
              ? 'bg-gradient-to-r from-pink-400/20 to-orange-400/20' 
              : 'bg-gradient-to-r from-pink-400/35 to-orange-400/35'
          }`}
          style={{
            top: '20%',
            left: '60%',
          }}
          animate={{
            x: [0, -100, 0],
            y: [0, 60, 0],
            rotate: [0, 180, 360],
          }}
          transition={{
            duration: 50,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        <motion.div
          className={`absolute w-48 h-48 rounded-full blur-lg ${
            isDarkMode 
              ? 'bg-gradient-to-r from-emerald-400/20 to-teal-400/20' 
              : 'bg-gradient-to-r from-emerald-400/35 to-teal-400/35'
          }`}
          style={{
            top: '80%',
            right: '5%',
          }}
          animate={{
            x: [0, 80, 0],
            y: [0, -60, 0],
            scale: [1, 1.1, 1],
          }}
          transition={{
            duration: 55,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* Subtle floating particles */}
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className={`absolute w-1 h-1 rounded-full ${
              isDarkMode ? 'bg-white/30' : 'bg-blue-500/20'
            }`}
            style={{
              left: `${5 + Math.random() * 90}%`,
              top: `${5 + Math.random() * 90}%`,
            }}
            animate={{
              y: [0, -40, 0],
              opacity: [0.3, 0.8, 0.3],
            }}
            transition={{
              duration: 25 + Math.random() * 15,
              repeat: Infinity,
              delay: Math.random() * 15,
              ease: "easeInOut",
            }}
          />
        ))}

        {/* Subtle grid pattern */}
        <div className={`absolute inset-0 opacity-3 ${
          isDarkMode 
            ? 'bg-[radial-gradient(circle_at_1px_1px,rgba(255,255,255,0.1)_1px,transparent_0)]'
            : 'bg-[radial-gradient(circle_at_1px_1px,rgba(0,0,0,0.05)_1px,transparent_0)]'
        } bg-[length:30px_30px]`} />
      </div>

                   {/* Theme Toggle - Bottom Right */}
             <motion.button
               initial={{ opacity: 0, scale: 0 }}
               animate={{ opacity: 1, scale: 1 }}
               onClick={toggleTheme}
               className={`fixed bottom-6 right-6 z-50 p-3 rounded-full backdrop-blur-sm border transition-all duration-300 ${
                 isDarkMode
                   ? 'bg-white/10 border-white/20 text-white hover:bg-white/20'
                   : 'bg-black/10 border-black/20 text-gray-900 hover:bg-black/20'
               }`}
               whileHover={{ scale: 1.1 }}
               whileTap={{ scale: 0.9 }}
             >
               {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
             </motion.button>



      {/* Hero Section */}
      <section id="home" className="relative z-10 text-center px-6 py-20 pt-32">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h1 className="text-6xl md:text-8xl font-bold mb-6">
            <span className={`bg-gradient-to-r from-blue-600 to-green-600 bg-clip-text text-transparent ${
              isDarkMode ? 'from-blue-400 to-green-400' : 'from-blue-600 to-green-600'
            }`}>
              Next-Generation
            </span>
            <br />
            <span className={`bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent ${
              isDarkMode ? 'from-purple-400 to-pink-400' : 'from-purple-600 to-pink-600'
            }`}>
              Health{' '}
              <span className="inline-block ml-2">
                {typingText}
                <motion.span
                  animate={{ opacity: [1, 0, 1] }}
                  transition={{ 
                    duration: 0.8,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                  className="inline-block w-0.5 h-6 bg-current ml-1"
                />
              </span>
            </span>
          </h1>
          <h2 className={`text-2xl md:text-3xl font-semibold mb-8 ${
            isDarkMode ? 'text-white/80' : 'text-gray-700'
          }`}>
            New Era of Medical Care
          </h2>
          <p className={`text-xl max-w-3xl mx-auto mb-12 ${
            isDarkMode ? 'text-white/70' : 'text-gray-600'
          }`}>
            Revolutionary healthcare management system with cryptographic transparency and AI-driven insights. 
            Every medical transaction and patient care activity is securely recorded and monitored in real-time.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onLogin}
              className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-10 py-5 rounded-full text-xl font-semibold hover:shadow-2xl transition-all duration-300 flex items-center gap-3"
            >
              Get Started <ArrowRight size={24} />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onLogin}
              className={`px-8 py-5 rounded-full text-xl font-semibold transition-all duration-300 border-2 ${
                isDarkMode 
                  ? 'text-white border-white/30 hover:bg-white/10 hover:border-white/50' 
                  : 'text-gray-700 border-gray-300 hover:bg-gray-50 hover:border-gray-400'
              }`}
            >
              Login
            </motion.button>
          </div>
        </motion.div>

        {/* Stats */}
        <motion.div
          id="stats"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.8 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-8 mt-20 max-w-4xl mx-auto"
        >
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 + index * 0.1 }}
              className={`text-center p-6 rounded-2xl backdrop-blur-sm border ${
                isDarkMode 
                  ? 'bg-white/5 border-white/10' 
                  : 'bg-white/50 border-gray-200 shadow-lg'
              }`}
            >
              <stat.icon className={`w-8 h-8 mx-auto mb-3 ${
                isDarkMode ? 'text-blue-400' : 'text-blue-600'
              }`} />
              <div className={`text-2xl font-bold mb-1 ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>{stat.value}</div>
              <div className={`text-sm ${
                isDarkMode ? 'text-white/70' : 'text-gray-600'
              }`}>{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative z-10 px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className={`text-4xl font-bold mb-4 ${
            isDarkMode ? 'text-white' : 'text-gray-900'
          }`}>Why Choose MedCare?</h2>
          <p className={`text-xl max-w-2xl mx-auto ${
            isDarkMode ? 'text-white/70' : 'text-gray-600'
          }`}>
            Experience the future of healthcare inventory management with cutting-edge security and transparency
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1, duration: 0.6 }}
              viewport={{ once: true }}
              className={`p-8 rounded-2xl backdrop-blur-sm border transition-all duration-300 hover:scale-105 ${
                isDarkMode 
                  ? 'bg-white/5 border-white/10 hover:bg-white/10' 
                  : 'bg-white/50 border-gray-200 shadow-lg hover:shadow-xl'
              }`}
            >
              <feature.icon className={`w-12 h-12 mb-6 ${
                isDarkMode ? 'text-blue-400' : 'text-blue-600'
              }`} />
              <h3 className={`text-xl font-semibold mb-4 ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>{feature.title}</h3>
              <p className={`${
                isDarkMode ? 'text-white/70' : 'text-gray-600'
              }`}>{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Testimonials Section */}
      <section id="testimonials" className="relative z-10 px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className={`text-4xl font-bold mb-4 ${
            isDarkMode ? 'text-white' : 'text-gray-900'
          }`}>Trusted by Healthcare Leaders</h2>
          <p className={`text-xl max-w-2xl mx-auto ${
            isDarkMode ? 'text-white/70' : 'text-gray-600'
          }`}>
            See what healthcare professionals are saying about MedCare
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={testimonial.author}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.2, duration: 0.6 }}
              viewport={{ once: true }}
              className={`p-8 rounded-2xl backdrop-blur-sm border ${
                isDarkMode 
                  ? 'bg-white/5 border-white/10' 
                  : 'bg-white/50 border-gray-200 shadow-lg'
              }`}
            >
              <p className={`text-lg mb-6 italic ${
                isDarkMode ? 'text-white/80' : 'text-gray-700'
              }`}>"{testimonial.quote}"</p>
              <div>
                <div className={`font-semibold ${
                  isDarkMode ? 'text-white' : 'text-gray-900'
                }`}>{testimonial.author}</div>
                <div className={`text-sm ${
                  isDarkMode ? 'text-white/60' : 'text-gray-500'
                }`}>{testimonial.role}</div>
                <div className={`text-sm ${
                  isDarkMode ? 'text-white/60' : 'text-gray-500'
                }`}>{testimonial.hospital}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section id="cta" className="relative z-10 px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className={`text-center max-w-4xl mx-auto p-12 rounded-3xl backdrop-blur-sm border ${
            isDarkMode 
              ? 'bg-white/5 border-white/10' 
              : 'bg-white/50 border-gray-200 shadow-xl'
          }`}
        >
          <h2 className={`text-4xl font-bold mb-6 ${
            isDarkMode ? 'text-white' : 'text-gray-900'
          }`}>Ready to Transform Your Healthcare Inventory?</h2>
          <p className={`text-xl mb-8 ${
            isDarkMode ? 'text-white/70' : 'text-gray-600'
          }`}>
            Join leading healthcare facilities worldwide in adopting next-generation inventory management. 
            Experience the future of secure, transparent, and automated healthcare supply chain management.
          </p>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onLogin}
            className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-full text-lg font-semibold hover:shadow-2xl transition-all duration-300 flex items-center gap-2 mx-auto"
          >
            Start Free Trial <ArrowRight size={20} />
          </motion.button>
        </motion.div>
      </section>
    </div>
  );
};

export default LandingPage;
